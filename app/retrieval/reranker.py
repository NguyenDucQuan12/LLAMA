from __future__ import annotations

"""

MỤC ĐÍCH
--------
Rerank các document đã được Qdrant tìm thấy.

Pipeline thường là:

    Câu hỏi
        -> embedding query
        -> Qdrant lấy top 20 nhanh bằng bi-encoder
        -> CrossEncoder đọc từng cặp [question, document]
        -> chấm lại độ liên quan
        -> kết hợp reranker score và dense score
        -> loại bớt đoạn trùng
        -> lấy top 5 gửi cho Llama

VÌ SAO KHÔNG DÙNG CROSS-ENCODER CHO TOÀN BỘ QDRANT?
---------------------------------------------------
Cross-encoder phải chạy model cho từng cặp:

    [question, document_1]
    [question, document_2]
    ...

Nó thường chính xác hơn dense retrieval, nhưng chậm hơn nhiều.
Vì vậy production thường:

    dense retrieval: lấy nhanh 20-100 ứng viên
    cross-encoder: chấm lại tập ứng viên nhỏ
    Llama: chỉ nhận top 5-10

CÁCH CHẠY TEST
--------------
Đặt file tại:

    app/retrieval/cross_encoder_reranker.py

Chạy từ thư mục gốc project:

1. Unit test, không tải model và không cần Internet:

    python3 -m app.retrieval.cross_encoder_reranker \
        --mode unit

2. Test bằng model thật trong Settings:

    python3 -m app.retrieval.cross_encoder_reranker \
        --mode model

LƯU Ý
-----
- `--mode model` có thể tải model từ Hugging Face trong lần đầu.
- Hãy tạo service trong cùng event loop/lifespan của FastAPI.
- Không dùng cùng object qua nhiều lần `asyncio.run()`.
"""

# argparse: đọc tham số --mode unit hoặc --mode model.
import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import torch
from sentence_transformers import CrossEncoder

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import Settings, get_settings
from retrieval.models import RetrievedChunk, RerankedChunk


# Logger dùng tên module hiện tại.
logger = logging.getLogger(__name__)


class CrossEncoderDocumentReranker:
    """
    Rerank document bằng CrossEncoder và lazy-load model.

    Lazy-load nghĩa là:
    - tạo object reranker chưa tải model;
    - request đầu tiên mới tải model;
    - các request sau dùng lại cùng model.

    Điều này giúp ứng dụng khởi động nhanh hơn và không tải model khi
    reranker đang bị tắt.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Lưu Settings và tạo các primitive điều phối async.

        Không tải model tại đây vì:
        - constructor nên nhanh;
        - model có thể nặng;
        - reranker có thể bị tắt;
        - ứng dụng có thể chỉ dùng endpoint không cần retrieval.
        """

        # Lưu toàn bộ cấu hình.
        self.settings = settings

        # Ban đầu chưa có model.
        # Sau lần _get_or_load_model() đầu tiên, field này chứa CrossEncoder.
        self.model: CrossEncoder | None = None

        # Chỉ một coroutine được phép tải model.
        # Nếu 10 request đầu tiên tới cùng lúc, chỉ request đầu tải;
        # 9 request còn lại chờ rồi dùng chung model đã tải.
        self.model_load_lock = asyncio.Lock()

        # Đọc và kiểm tra cấu hình ngay từ đầu.
        # Nếu batch_size=0 hoặc trọng số sai, báo lỗi trước khi tải model.
        self._validate_configuration()

        # Giới hạn số lời gọi model.predict chạy đồng thời.
        #
        # Lý do:
        # - mỗi predict có thể dùng nhiều RAM/VRAM;
        # - nhiều request đồng thời có thể gây Out Of Memory;
        # - MPS thường ổn định hơn khi concurrency thấp.
        maximum_concurrency = getattr(self.settings, "reranker_max_concurrency", 1)

        # _validate_configuration() đã xác nhận đây là int > 0.
        self.predict_semaphore = asyncio.Semaphore(
            maximum_concurrency
        )

    async def rerank(self, question: str, candidates: Sequence[RetrievedChunk], top_k: int | None = None) -> list[RerankedChunk]:
        """
        Chấm lại ứng viên và trả top K tốt nhất.

        Parameters
        ----------
        question:
            Câu hỏi nguyên bản của người dùng.

        candidates:
            Các chunk Qdrant đã tìm thấy, thường là top 20.

        top_k:
            Số chunk cuối cần lấy.
            Nếu None, dùng settings.rerank_top_k.

        Returns
        -------
        Danh sách RerankedChunk đã sắp xếp giảm dần theo final_score.
        """

        # ----------------------------------------------------
        # BƯỚC 1: CHUẨN HÓA CÂU HỎI VÀ TOP K
        # ----------------------------------------------------
        if not isinstance(question, str):
            raise TypeError("Câu hỏi phải là chuỗi văn bản.")
        
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Câu hỏi không được để trống.")
        
        # Lấy ra số câu hỏi cuối cùng để đưa vào llama
        selected_top_k = (
            top_k
            if top_k is not None
            else self.settings.rerank_top_k
        )

        if (isinstance(selected_top_k, bool) or not isinstance(selected_top_k, int)):
            raise TypeError("Số chunk đưa vào llama phải là số nguyên.")

        if selected_top_k <= 0:
            raise ValueError("Số chunk đưa vào llama phải lớn hơn 0.")

        # Không có ứng viên thì không cần tải/gọi model.
        if not candidates:
            return []

        # ----------------------------------------------------
        # BƯỚC 2: KIỂM TRA VÀ LOẠI DOCUMENT TRÙNG
        # ----------------------------------------------------
        validated_candidates = self._validate_and_prepare_candidates(candidates)

        # Nếu mọi candidate rỗng thì trả về rỗng
        if not validated_candidates:
            return []

        # Xoá các bản candidate trùng nhau
        deduplicated_candidates = self._remove_exact_duplicates(validated_candidates)

        # Nếu danh sách candidate ít hơn số lượng kỳ vọng thì chỉ lấy đúng số cánisate tìm được
        selected_top_k = min(selected_top_k, len(deduplicated_candidates))

        # ----------------------------------------------------
        # BƯỚC 3: FALLBACK NGAY NẾU RERANKER BỊ TẮT
        # ----------------------------------------------------

        if not self.settings.reranker_enabled:
            return self._dense_fallback(candidates=deduplicated_candidates, top_k=selected_top_k)

        try:
            # -----------------------------------------------
            # BƯỚC 4: LAZY-LOAD MODEL
            # -----------------------------------------------
            model = await self._get_or_load_model()

            # -----------------------------------------------
            # BƯỚC 5: CHẤM CROSS-ENCODER
            # -----------------------------------------------
            # Semaphore ngăn quá nhiều predict đồng thời.
            async with self.predict_semaphore:
                # CrossEncoder.predict là hàm đồng bộ. Nếu gọi trực tiếp có thể gây nghẽn
                # to_thread chuyển lời gọi sang worker thread để event loop vẫn tiếp nhận request khác.
                reranker_scores = await asyncio.to_thread(
                    self._predict_scores,
                    model, normalized_question, deduplicated_candidates,
                )

            if len(reranker_scores) != len(deduplicated_candidates):
                raise RuntimeError(
                    "Số reranker score không khớp candidate: "
                    f"{len(reranker_scores)} != "
                    f"{len(deduplicated_candidates)}."
                )

            # -----------------------------------------------
            # BƯỚC 6: KẾT HỢP RERANKER VÀ DENSE SCORE
            # -----------------------------------------------

            reranker_weight, dense_weight = self._get_normalized_score_weights()

            scored_candidates: list[RerankedChunk] = []

            # strict=True ngăn việc âm thầm bỏ candidate nếu hai list có độ dài khác nhau.
            for candidate, reranker_score in zip(deduplicated_candidates, reranker_scores, strict=True):
                normalized_dense_score = self._normalize_dense_score(candidate.dense_score)

                # Khi hai trọng số đã được chuẩn hóa và cả hai score
                # trong [0,1], final_score cũng nằm trong [0,1].
                final_score = (
                    reranker_weight
                    * reranker_score
                    + dense_weight
                    * normalized_dense_score
                )

                candidate_data = self._candidate_to_dictionary(candidate)

                # Bảo vệ trường hợp object đầu vào đã có hai field này.
                candidate_data.pop("reranker_score", None)
                candidate_data.pop("final_score", None)

                scored_candidates.append(
                    RerankedChunk(
                        **candidate_data,
                        reranker_score=reranker_score,
                        final_score=float(final_score),
                    )
                )

            # -----------------------------------------------
            # BƯỚC 7: SẮP XẾP GIẢM DẦN
            # -----------------------------------------------

            scored_candidates.sort(key=lambda item: item.final_score, reverse=True)

            # -----------------------------------------------
            # BƯỚC 8: CHỌN TOP K 
            # -----------------------------------------------

            return self._select_diverse_top_k(candidates=scored_candidates, top_k=selected_top_k)

        except Exception:
            # Không fallback nếu cấu hình yêu cầu fail-fast.
            if not (self.settings.reranker_allow_dense_fallback):
                raise

            # logger.exception in cả traceback.
            logger.exception(
                "Reranker thất bại. "
                "Hệ thống tạm dùng dense score Qdrant."
            )

            return self._dense_fallback(candidates=deduplicated_candidates, top_k=selected_top_k)

    async def _get_or_load_model(self) -> CrossEncoder:
        """
        Tải model đúng một lần cho mỗi object/process.

        Đây là double-checked locking:

        1. Kiểm tra ngoài lock để request sau trả nhanh.
        2. Nếu model chưa có, vào lock.
        3. Kiểm tra lần nữa vì coroutine khác có thể vừa tải xong.
        4. Tải model.
        """

        # Fast path: model đã được tải.
        if self.model is not None:
            return self.model

        # Chỉ một coroutine được vào vùng tải model.
        async with self.model_load_lock:
            # Kiểm tra lại sau khi giành được lock.
            if self.model is not None:
                return self.model
            # Chọn phần cứng phù hợp
            selected_device = self._resolve_device()

            logger.info("Đang tải reranker `%s` trên device `%s`.", self.settings.reranker_model_name, selected_device)

            # Tải model là synchronous và có thể chậm do đọc disk/download.
            self.model = await asyncio.to_thread(self._create_model_sync, selected_device)

            return self.model

    def _create_model_sync(self, selected_device: str) -> CrossEncoder:
        """
        Tạo CrossEncoder đồng bộ.

        Method riêng này giúp unit test override để trả fake model,
        không cần download model thật.
        """
        # Lấy độ dài tối đa input của model
        # Question: 40 token
        # Document: 500 token
        # Special tokens: vài token
        # Tổng: hơn 512, Phần thừa bị cắt bỏ
        maximum_length = getattr(self.settings, "reranker_max_length", 512)

        # _validate_configuration() đã xác nhận đây là int > 0.

        apply_sigmoid = bool(getattr(self.settings, "reranker_apply_sigmoid", True))

        # Sigmoid:
        #   logit bất kỳ -> quy đổi giá trị về khoảng score 0 đến 1.
        #
        # Identity:
        #   giữ raw logit. Khi đó không được giả định score nằm trong 0-1.
        #
        # Pipeline này kết hợp score với dense score 0-1,
        # vì vậy mặc định nên dùng Sigmoid.
        activation_function = (
            torch.nn.Sigmoid()
            if apply_sigmoid
            else torch.nn.Identity()
        )

        return CrossEncoder(
            self.settings.reranker_model_name,
            device=selected_device,
            max_length=maximum_length,
            activation_fn=activation_function,
        )

    def _predict_scores(self, model: CrossEncoder, question: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        """
        Tạo các cặp [question, document] và trả scalar score.

        Một input của CrossEncoder gồm cả hai text:

            (question, contextualized_text)

        `max_length` là tổng token của cả cặp sau khi tokenizer ghép chúng,
        không phải 512 token riêng cho question và 512 riêng cho document.
        """

        sentence_pairs: list[tuple[str, str]] = []
        # Đối với từng candidates được lấy từ qdrant, ghép nó với câu hỏi và xử lý
        for candidate in candidates:
            # Ghép 1 cặp câu hỏi và contextualizer
            sentence_pairs.append(
                (
                    question,
                    candidate.contextualized_text,
                )
            )
        # Model xử lý theo từng batch. Tăng batch tuỳ theo cấu hình máy tính
        # Sau đó trả về danh sách kết quả chứa điểm số của từng cặp
        prediction_result = model.predict(
            sentence_pairs,
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Một số phiên bản/model trả numpy array.
        # Chuyển thành iterable Python ổn định.
        try:
            raw_items = list(prediction_result)
        except TypeError as exception:
            raise RuntimeError(
                "CrossEncoder.predict không trả một sequence score."
            ) from exception

        flattened_scores: list[float] = []

        for score_index, item in enumerate(raw_items):
            score = self._extract_scalar_score(item=item, score_index=score_index)

            if not math.isfinite(score):
                raise RuntimeError(
                    f"Reranker score {score_index} "
                    "là NaN hoặc Infinity."
                )

            # Pipeline mặc định dùng Sigmoid để score nằm trong [0,1].
            if bool(getattr(self.settings,"reranker_apply_sigmoid",True,)):
                # Cho phép sai số số thực rất nhỏ.
                if score < -1e-6 or score > 1.000001:
                    raise RuntimeError(
                        f"Reranker score {score_index}={score} "
                        "nằm ngoài [0,1] dù đang bật Sigmoid."
                    )

                score = min(1.0, max(0.0, score))

            flattened_scores.append(score)

        return flattened_scores

    def _extract_scalar_score(self, item: Any, score_index: int) -> float:
        """
        Chuyển một output score thành float scalar.

        Hỗ trợ:
        - float/int;
        - numpy scalar;
        - tensor scalar;
        - list/tuple có đúng một phần tử.

        Nếu model trả nhiều nhãn cho một document, hàm báo lỗi thay vì
        tùy ý chọn nhãn đầu tiên.
        """

        if isinstance(item, bool):
            raise TypeError(f"Reranker score {score_index} là boolean.")

        if isinstance(item, (int, float)):
            return float(item)

        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise RuntimeError(f"Reranker score {score_index} có "f"{len(item)} nhãn; pipeline cần một scalar.")

            return self._extract_scalar_score(item[0],score_index,)

        item_method = getattr(item, "item", None)

        if callable(item_method):
            try:
                scalar_value = item_method()
            except (ValueError, RuntimeError) as exception:
                raise RuntimeError(
                    f"Reranker output {score_index} không phải scalar. "
                    "Model có thể là classifier nhiều nhãn."
                ) from exception

            if isinstance(
                scalar_value,
                (int, float),
            ) and not isinstance(
                scalar_value,
                bool,
            ):
                return float(scalar_value)

        raise TypeError(
            f"Không chuyển được reranker score {score_index} "
            f"từ kiểu {type(item).__name__} thành float."
        )

    def _resolve_device(self) -> str:
        """
        Chọn CUDA, MPS hoặc CPU.

        Cấu hình hợp lệ:
        - auto
        - cpu
        - mps
        - cuda
        - cuda:0, cuda:1...
        """
        # Lấy cấu hình phần cứng từ biến môi trường
        configured_device = str(self.settings.reranker_device).strip().lower()

        if not configured_device:
            raise ValueError("reranker_device không được rỗng.")

        if configured_device == "auto":
            # Ưu tiên NVIDIA CUDA nếu có.
            if torch.cuda.is_available():
                return "cuda"

            # Trên Apple Silicon, dùng Metal/MPS nếu khả dụng.
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return "mps"

            return "cpu"

        if configured_device == "cpu":
            return "cpu"

        if configured_device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("reranker_device='mps' nhưng MPS không khả dụng.")

            return "mps"

        if (
            configured_device == "cuda"
            or re.fullmatch(r"cuda:\d+", configured_device)
        ):
            if not torch.cuda.is_available():
                raise RuntimeError("reranker_device dùng CUDA nhưng CUDA không khả dụng.")

            return configured_device

        raise ValueError(
            "reranker_device chỉ hỗ trợ "
            "auto, cpu, mps, cuda hoặc cuda:N."
        )

    def _normalize_dense_score(self, dense_score: float) -> float:
        """
        Chuyển cosine similarity từ [-1,1] sang [0,1].

        Công thức:
            normalized = (dense_score + 1) / 2

        Ví dụ:
            -1.0 -> 0.0
             0.0 -> 0.5
             1.0 -> 1.0

        Chỉ dùng công thức này nếu Qdrant collection sử dụng COSINE.
        """

        if isinstance(dense_score, bool):
            raise TypeError("dense_score không được là boolean.")

        try:
            numeric_score = float(dense_score)
        except (TypeError, ValueError) as exception:
            raise TypeError("dense_score phải chuyển được thành float.") from exception

        if not math.isfinite(numeric_score):
            raise ValueError("dense_score là NaN hoặc Infinity. Giá trị không hợp lệ")

        normalized_score = (numeric_score + 1.0) / 2.0

        # Clamp để xử lý sai số rất nhỏ hoặc score ngoài khoảng dự kiến.
        return min(1.0, max(0.0, normalized_score))

    def _get_normalized_score_weights(self) -> tuple[float, float]:
        """
        Chuẩn hóa hai trọng số để tổng bằng 1.

        Ví dụ cấu hình:
            reranker_score_weight = 8
            dense_score_weight = 2

        Sau chuẩn hóa:
            0.8 và 0.2

        Nhờ vậy người vận hành không bắt buộc nhập đúng tổng 1.
        """

        reranker_weight = float(self.settings.reranker_score_weight)
        dense_weight = float(self.settings.dense_score_weight)

        if not math.isfinite(reranker_weight) or not math.isfinite(dense_weight):
            raise ValueError("Các score weight phải là số hữu hạn.")

        if (reranker_weight < 0 or dense_weight < 0):
            raise ValueError("Các score weight không được âm.")

        total_weight = (reranker_weight + dense_weight)

        if total_weight <= 0:
            raise ValueError("Tổng score weight phải lớn hơn 0.")

        return (
            reranker_weight / total_weight,
            dense_weight / total_weight,
        )

    def _validate_and_prepare_candidates(self, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Kiểm tra candidate trước khi gọi model.

        Candidate có text rỗng bị bỏ qua vì CrossEncoder không nhận được
        thông tin có ích từ một document rỗng.

        dense_score sai kiểu/NaN làm pipeline dừng vì đó là lỗi dữ liệu.
        """

        output: list[RetrievedChunk] = []
        # Xử lý toàn bộ chunk tìm được trong adrant
        for index, candidate in enumerate(candidates):
            # LẤY Ccontextualize_text
            contextualized_text = getattr(candidate, "contextualized_text", None)

            if not isinstance(contextualized_text, str):
                raise TypeError(f"Candidate {index} thiếu contextualized_text dạng string.")

            # Chuẩn hoá context
            normalized_text = contextualized_text.strip()

            if not normalized_text:
                logger.warning("Bỏ candidate %s vì văn bản rỗng.", index)
                continue
            
            # LẤy điểm số của candidate này
            dense_score = getattr(candidate, "dense_score", None)

            # Gọi hàm normalize để kiểm tra score,
            # nhưng không thay đổi object đầu vào.
            self._normalize_dense_score(dense_score)

            # Lấy thông tin định danh của candidate
            point_id = getattr(candidate, "point_id", None)
            if point_id is None or not str(point_id).strip():
                raise ValueError(f"Candidate {index} thiếu point_id.")

            # Nếu hợp lệ thì thêm nó vào danh sách cuối cùng
            output.append(candidate)

        return output

    def _remove_exact_duplicates(self, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Loại các chunk có nội dung giống nhau sau chuẩn hóa.

        Nếu hai chunk giống nội dung, giữ chunk có dense_score cao hơn,
        thay vì giữ tùy ý chunk xuất hiện đầu tiên.
        """

        best_by_normalized_text: dict[str, RetrievedChunk,] = {}
        # Duyệt từng chuỗi contextualized_text của candidate, chuẩn hóa và loại bỏ trùng lặp
        for candidate in candidates:
            # Chuẩn hóa để kiểm tra trùng chính xác.
            normalized_text = (self._normalize_for_duplicate_check(candidate.contextualized_text))
            # Thử lấy candidate đã có trong dict theo normalized_text
            existing = best_by_normalized_text.get(normalized_text)

            # Nếu existing là None nghĩa là chưa có candidate nào trùng normalized_text, thì thêm candidate vào dict
            if existing is None:
                best_by_normalized_text[normalized_text] = candidate
                continue
            
            # Nếu existing đã có, so sánh dense_score của candidate hiện tại với existing, giữ candidate có dense_score cao hơn
            if (float(candidate.dense_score) > float(existing.dense_score)):
                best_by_normalized_text[normalized_text] = candidate

        # Chuyển dict thành list, giữ thứ tự dense giảm dần trước khi rerank/fallback.
        unique_candidates = list(best_by_normalized_text.values())

        # Sắp xếp theo dense_score giảm dần để top K đầu tiên là các candidate có dense_score cao nhất.
        unique_candidates.sort(
            key=lambda item: float(item.dense_score),
            reverse=True,
        )

        return unique_candidates

    def _select_diverse_top_k(self, candidates: Sequence[RerankedChunk], top_k: int) -> list[RerankedChunk]:
        """
        Chọn top K nhưng ưu tiên nội dung đa dạng.

        Ví dụ:
            D1: cách gọi robot
            D2: cách gọi robot, câu chữ gần giống D1
            D3: lỗi robot không chạy

        Nếu D1 và D2 quá giống nhau, ưu tiên D1 + D3 trước.
        """

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        # Lấy ngưỡng giá trị để làm mốc
        threshold = float(self.settings.reranker_duplicate_jaccard_threshold)

        if not 0 <= threshold <= 1:
            raise ValueError("reranker_duplicate_jaccard_threshold phải nằm trong [0,1].")

        selected_candidates: list[RerankedChunk] = []
        selected_point_ids: set[str] = set()

        for candidate in candidates:
            # Nếu đã đủ số lượng thì dừng lại
            if len(selected_candidates) >= top_k:
                break
            
            # Lấy id của candidate này
            point_id = str(candidate.point_id)

            # Bỏ point ID trùng.
            if point_id in selected_point_ids:
                continue

            is_near_duplicate = False

            for selected_candidate in (selected_candidates):
                similarity = (
                    self._token_set_jaccard_similarity(candidate.contextualized_text, selected_candidate.contextualized_text)
                )
                # Nếu giá trị similarity >= threshold, coi như candidate này trùng với selected_candidate, bỏ qua candidate này
                if similarity >= threshold:
                    is_near_duplicate = True
                    break
            
            # Nếu không trùng thì thêm nó vào danh sách
            if not is_near_duplicate:
                selected_candidates.append(candidate)
                selected_point_ids.add(point_id)

        # Nếu diversity làm thiếu top K, bổ sung theo final_score.
        # Như vậy caller vẫn nhận đủ top K khi có đủ candidate.
        if len(selected_candidates) < top_k:
            for candidate in candidates:
                if (len(selected_candidates) >= top_k):
                    break

                point_id = str(candidate.point_id)

                if point_id in selected_point_ids:
                    continue

                selected_candidates.append(candidate)
                selected_point_ids.add(point_id)

        return selected_candidates

    def _dense_fallback(self, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RerankedChunk]:
        """
        Dùng dense score nếu reranker bị tắt hoặc gặp lỗi.

        Fallback vẫn:
        - chuẩn hóa dense score;
        - chuyển sang RerankedChunk;
        - áp dụng diversity.
        """
        # Sắp xếp candidates theo dense_score giảm dần để top K đầu tiên là các candidate có dense_score cao nhất.
        sorted_candidates = sorted(
            candidates,
            key=lambda item: float(item.dense_score),
            reverse=True,
        )

        output: list[RerankedChunk] = []

        for candidate in sorted_candidates:
            candidate_data = self._candidate_to_dictionary(candidate)

            candidate_data.pop("reranker_score",None,)
            candidate_data.pop("final_score",None,)

            output.append(
                RerankedChunk(
                    **candidate_data,
                    reranker_score=None,
                    final_score=(
                        self._normalize_dense_score(
                            candidate.dense_score
                        )
                    ),
                )
            )

        return self._select_diverse_top_k(
            candidates=output,
            top_k=min(top_k,len(output),),
        )

    def _candidate_to_dictionary(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        """
        Chuyển candidate Pydantic thành dictionary.

        Hỗ trợ:
        - Pydantic v2: model_dump()
        - Pydantic v1: dict()
        - fake object test có model_dump()
        """

        model_dump_method = getattr(
            candidate,
            "model_dump",
            None,
        )

        if callable(model_dump_method):
            data = model_dump_method()

            if isinstance(data, dict):
                return dict(data)

        dict_method = getattr(
            candidate,
            "dict",
            None,
        )

        if callable(dict_method):
            data = dict_method()

            if isinstance(data, dict):
                return dict(data)

        raise TypeError(
            "Candidate phải hỗ trợ model_dump() hoặc dict()."
        )

    def _normalize_for_duplicate_check(
        self,
        text: str,
    ) -> str:
        """
        Chuẩn hóa để kiểm tra trùng chính xác.

        Ví dụ:
            "GỌI   ROBOT\n"
            "gọi robot"

        Cả hai thành:
            "gọi robot"
        """

        normalized_text = text.lower()

        # Gộp mọi khoảng trắng liên tiếp thành một dấu cách.
        normalized_text = re.sub(
            r"\s+",
            " ",
            normalized_text,
        )

        return normalized_text.strip()

    def _token_set_jaccard_similarity(self,first_text: str,second_text: str,) -> float:
        """
        Tính Jaccard similarity giữa tập token.

        Công thức:
            |A giao B| / |A hợp B|

        Ví dụ:
            A = {"gọi", "robot"}
            B = {"gọi", "robot", "nhận", "hàng"}

            giao = 2
            hợp = 4
            similarity = 0.5
        
        Tuy nhiên nếu có từ đồng nghĩa như: 
            A = {"gọi", "robot"}
            B = {"nhấn", "nút", "call", "robot"}
        thì similarity = 0.25, nhưng hàm hiện tại không nhận biết được từ đồng nghĩa.  
        Nếu muốn nhận biết từ đồng nghĩa, cần dùng embedding hoặc model NLP để so sánh semantic similarity.
        """
        # Ta sử dụng regex để tách token, bỏ qua các ký tự đặc biệt và chỉ giữ lại từ và số.
        first_tokens = set(re.findall(r"[\w-]+", first_text.lower(), flags=re.UNICODE))
        second_tokens = set(re.findall(r"[\w-]+", second_text.lower(), flags=re.UNICODE))

        if (not first_tokens and not second_tokens):
            return 1.0

        union = first_tokens | second_tokens
        if not union:
            return 0.0

        intersection = (first_tokens & second_tokens)

        return len(intersection) / len(union)

    def _validate_configuration(self) -> None:
        """
        Kiểm tra các Settings mà class sử dụng.

        Dùng getattr với default cho các field mới để không bắt buộc
        phải sửa Settings ngay lập tức.
        """

        model_name = str(
            self.settings.reranker_model_name
        ).strip()

        if not model_name:
            raise ValueError(
                "reranker_model_name không được rỗng."
            )

        integer_fields = {
            "rerank_top_k": (
                self.settings.rerank_top_k
            ),
            "reranker_batch_size": (
                self.settings.reranker_batch_size
            ),
            "reranker_max_length": getattr(
                self.settings,
                "reranker_max_length",
                512,
            ),
            "reranker_max_concurrency": getattr(
                self.settings,
                "reranker_max_concurrency",
                1,
            ),
        }

        for field_name, value in (
            integer_fields.items()
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{field_name} phải là int."
                )

            if value <= 0:
                raise ValueError(
                    f"{field_name} phải lớn hơn 0."
                )

        # Gọi để validation trọng số chạy.
        self._get_normalized_score_weights()

        threshold = float(
            self.settings
            .reranker_duplicate_jaccard_threshold
        )

        if (
            not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "reranker_duplicate_jaccard_threshold "
                "phải là số trong [0,1]."
            )


# ============================================================
# UNIT TEST KHÔNG CẦN MODEL THẬT
# ============================================================

@dataclass
class _TestSettings:
    """
    Settings giả chỉ chứa field reranker sử dụng.
    """

    rerank_top_k: int = 3
    reranker_enabled: bool = True
    reranker_allow_dense_fallback: bool = True

    reranker_model_name: str = "fake-model"
    reranker_batch_size: int = 4
    reranker_device: str = "cpu"
    reranker_max_length: int = 512
    reranker_max_concurrency: int = 1
    reranker_apply_sigmoid: bool = True

    reranker_score_weight: float = 8.0
    dense_score_weight: float = 2.0

    reranker_duplicate_jaccard_threshold: float = 0.55


@dataclass
class _TestRetrievedChunk:
    """
    Candidate giả có đúng các field class reranker sử dụng.
    """

    point_id: str
    contextualized_text: str
    dense_score: float
    source_file: str = "test.docx"
    chunk_index: int = 0

    def model_dump(self) -> dict[str, Any]:
        """
        Mô phỏng Pydantic v2 model_dump().
        """

        return {
            "point_id": self.point_id,
            "contextualized_text": (
                self.contextualized_text
            ),
            "dense_score": self.dense_score,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
        }


class _TestRerankedChunk:
    """
    Mô phỏng RerankedChunk của project trong unit test.

    Nhận **kwargs để không phụ thuộc cấu trúc Pydantic thật.
    """

    def __init__(
        self,
        **data: Any,
    ) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


class _FakeCrossEncoder:
    """
    Fake model chấm điểm bằng từ khóa.

    Không tải model, không dùng GPU và chạy rất nhanh.
    """

    def predict(
        self,
        sentence_pairs: Sequence[
            tuple[str, str]
        ],
        **_: Any,
    ) -> list[float]:
        scores: list[float] = []

        for _, document in sentence_pairs:
            normalized = document.lower()

            if "nhấn nút call robot" in normalized:
                scores.append(0.96)
            elif "gửi nhiệm vụ gọi robot" in normalized:
                scores.append(0.90)
            elif "robot không chạy" in normalized:
                scores.append(0.78)
            elif "tồn kho" in normalized:
                scores.append(0.12)
            else:
                scores.append(0.30)

        return scores


class _UnitTestReranker(
    CrossEncoderDocumentReranker
):
    """
    Override đúng method tạo model để unit test không download.
    """

    def _create_model_sync(
        self,
        selected_device: str,
    ) -> CrossEncoder:
        del selected_device

        # Cast runtime không cần thiết;
        # Python dùng duck typing vì _predict_scores chỉ gọi predict().
        return _FakeCrossEncoder()  # type: ignore[return-value]


class _FailingCrossEncoder:
    """
    Fake model luôn lỗi để test dense fallback.
    """

    def predict(
        self,
        *_: Any,
        **__: Any,
    ) -> list[float]:
        raise RuntimeError(
            "Lỗi model giả phục vụ test."
        )


class _FailingTestReranker(
    CrossEncoderDocumentReranker
):
    def _create_model_sync(
        self,
        selected_device: str,
    ) -> CrossEncoder:
        del selected_device
        return _FailingCrossEncoder()  # type: ignore[return-value]


def _make_test_candidates() -> list[
    _TestRetrievedChunk
]:
    """
    Tạo candidate có:
    - một exact duplicate;
    - hai đoạn gần giống;
    - một dense result cao nhưng không liên quan;
    - một đoạn lỗi robot.
    """

    return [
        _TestRetrievedChunk(
            point_id="D1",
            contextualized_text=(
                "Quy trình gọi robot: "
                "kiểm tra pallet, chọn vị trí và "
                "nhấn nút Call Robot."
            ),
            dense_score=0.70,
            chunk_index=1,
        ),
        _TestRetrievedChunk(
            point_id="D2",
            contextualized_text=(
                "Báo cáo tồn kho văn phòng phẩm "
                "và số lượng bút còn lại."
            ),
            dense_score=0.95,
            chunk_index=2,
        ),
        _TestRetrievedChunk(
            point_id="D3",
            contextualized_text=(
                "Chọn vị trí, kiểm tra pallet và "
                "gửi nhiệm vụ gọi robot bằng nút Call Robot."
            ),
            dense_score=0.69,
            chunk_index=3,
        ),
        _TestRetrievedChunk(
            point_id="D4",
            # Exact duplicate D1 nhưng dense thấp hơn.
            contextualized_text=(
                "  QUY TRÌNH GỌI ROBOT: kiểm tra pallet, "
                "chọn vị trí và nhấn nút Call Robot.  "
            ),
            dense_score=0.50,
            chunk_index=4,
        ),
        _TestRetrievedChunk(
            point_id="D5",
            contextualized_text=(
                "Khi robot không chạy, kiểm tra trạng thái, "
                "nguồn điện và nhiệm vụ đang treo."
            ),
            dense_score=0.45,
            chunk_index=5,
        ),
    ]


async def test_unit_rerank() -> None:
    """
    Test luồng cross-encoder bằng fake model.
    """

    settings = _TestSettings()
    reranker = _UnitTestReranker(
        settings  # type: ignore[arg-type]
    )

    # RerankedChunk là import của project.
    # Unit test tạm thay bằng class giả để không phụ thuộc schema thật.
    original_reranked_chunk = globals()[
        "RerankedChunk"
    ]
    globals()["RerankedChunk"] = (
        _TestRerankedChunk
    )

    try:
        candidates = _make_test_candidates()

        result = await reranker.rerank(
            question=(
                "Làm thế nào để gọi robot "
                "và xử lý khi robot không chạy?"
            ),
            candidates=candidates,  # type: ignore[arg-type]
            top_k=3,
        )

        serialized = [
            item.model_dump()
            for item in result
        ]

        print("=" * 80)
        print("UNIT TEST: CROSS-ENCODER RERANK")
        print("=" * 80)
        print(
            json.dumps(
                serialized,
                ensure_ascii=False,
                indent=2,
            )
        )

        returned_ids = [
            item.point_id
            for item in result
        ]

        # D4 là exact duplicate có dense thấp hơn nên phải bị loại.
        assert "D4" not in returned_ids

        # D1 phải đứng đầu vì fake cross-encoder chấm cao.
        assert returned_ids[0] == "D1"

        # Đủ đúng top 3.
        assert len(result) == 3

        # final_score phải nằm trong [0,1].
        assert all(
            0 <= item.final_score <= 1
            for item in result
        )

        print(
            "\nKẾT QUẢ: unit rerank thành công."
        )

    finally:
        globals()["RerankedChunk"] = (
            original_reranked_chunk
        )


async def test_dense_disabled() -> None:
    """
    Test khi reranker_enabled=False.
    """

    settings = _TestSettings(
        reranker_enabled=False
    )

    reranker = _UnitTestReranker(
        settings  # type: ignore[arg-type]
    )

    original_reranked_chunk = globals()[
        "RerankedChunk"
    ]
    globals()["RerankedChunk"] = (
        _TestRerankedChunk
    )

    try:
        result = await reranker.rerank(
            question="Câu hỏi test",
            candidates=(  # type: ignore[arg-type]
                _make_test_candidates()
            ),
            top_k=2,
        )

        # D2 có dense_score cao nhất nên đứng đầu fallback.
        assert result[0].point_id == "D2"
        assert result[0].reranker_score is None

        print(
            "KẾT QUẢ: dense disabled fallback thành công."
        )

    finally:
        globals()["RerankedChunk"] = (
            original_reranked_chunk
        )


async def test_model_failure_fallback() -> None:
    """
    Test model lỗi nhưng cấu hình cho phép dense fallback.
    """

    settings = _TestSettings(
        reranker_allow_dense_fallback=True
    )

    reranker = _FailingTestReranker(
        settings  # type: ignore[arg-type]
    )

    original_reranked_chunk = globals()[
        "RerankedChunk"
    ]
    globals()["RerankedChunk"] = (
        _TestRerankedChunk
    )

    try:
        result = await reranker.rerank(
            question="Câu hỏi test",
            candidates=(  # type: ignore[arg-type]
                _make_test_candidates()
            ),
            top_k=2,
        )

        assert len(result) == 2
        assert all(
            item.reranker_score is None
            for item in result
        )

        print(
            "KẾT QUẢ: model failure fallback thành công."
        )

    finally:
        globals()["RerankedChunk"] = (
            original_reranked_chunk
        )


async def test_input_validation() -> None:
    """
    Test các lỗi input phải bị chặn.
    """

    reranker = _UnitTestReranker(
        _TestSettings()  # type: ignore[arg-type]
    )

    candidates = _make_test_candidates()

    try:
        await reranker.rerank(
            question="   ",
            candidates=candidates,  # type: ignore[arg-type]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Câu hỏi rỗng phải gây ValueError."
        )

    try:
        await reranker.rerank(
            question="Câu hỏi",
            candidates=candidates,  # type: ignore[arg-type]
            top_k=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "top_k=0 phải gây ValueError."
        )

    print(
        "KẾT QUẢ: input validation thành công."
    )


async def run_all_unit_tests() -> None:
    """
    Chạy tất cả test không cần model thật.
    """

    await test_unit_rerank()
    await test_dense_disabled()
    await test_model_failure_fallback()
    await test_input_validation()

    print()
    print("=" * 80)
    print("TOÀN BỘ UNIT TEST ĐÃ THÀNH CÔNG")
    print("=" * 80)


# ============================================================
# TEST MODEL THẬT
# ============================================================

async def test_real_model() -> None:
    """
    Dùng model và Settings thật của project.

    Test này có thể tải model trong lần đầu.
    """

    settings = get_settings()
    reranker = CrossEncoderDocumentReranker(
        settings
    )

    original_reranked_chunk = globals()[
        "RerankedChunk"
    ]
    globals()["RerankedChunk"] = (
        _TestRerankedChunk
    )

    try:
        candidates = _make_test_candidates()

        result = await reranker.rerank(
            question=(
                "Làm thế nào để gọi robot nhận hàng?"
            ),
            candidates=candidates,  # type: ignore[arg-type]
            top_k=3,
        )

        print("=" * 80)
        print("TEST MODEL THẬT")
        print("=" * 80)

        print(
            json.dumps(
                [
                    item.model_dump()
                    for item in result
                ],
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        globals()["RerankedChunk"] = (
            original_reranked_chunk
        )


# ============================================================
# CLI VÀ __main__
# ============================================================

def build_argument_parser() -> (
    argparse.ArgumentParser
):
    """
    Tạo command-line parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test CrossEncoderDocumentReranker."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "unit",
            "model",
        ],
        default="unit",
        help=(
            "unit: fake model, không download; "
            "model: dùng Settings và model thật."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
        default="INFO",
    )

    return parser


async def main() -> None:
    """
    Main coroutine duy nhất.

    Một chương trình CLI async chỉ nên gọi asyncio.run() một lần.
    """

    args = build_argument_parser().parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    if args.mode == "unit":
        await run_all_unit_tests()
        return

    if args.mode == "model":
        await test_real_model()
        return

    raise RuntimeError(
        f"Mode không hỗ trợ: {args.mode}"
    )


if __name__ == "__main__":
    try:
        # Chỉ tạo một event loop cho toàn bộ test.
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "Đã dừng bởi người dùng.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exception:
        logger.exception(
            "Test reranker thất bại."
        )

        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )

        raise SystemExit(1)