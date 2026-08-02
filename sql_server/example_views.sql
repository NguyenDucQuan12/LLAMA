/*
===============================================================================
CÁC VIEW MẪU DÀNH CHO RAG
===============================================================================

Mục đích:
- Không cho ứng dụng RAG truy cập trực tiếp quá nhiều bảng nghiệp vụ.
- Chỉ công khai các cột cần thiết thông qua view read-only.
- Tài khoản `rag_reader` chỉ được SELECT trên các view này.

QUAN TRỌNG:
- Tên bảng và cột bên dưới chỉ là ví dụ.
- Hãy thay `dbo.Your...Table` bằng bảng thật trong hệ thống WMS.
- Không chạy nguyên file này khi chưa chỉnh sửa.
*/

/*
-------------------------------------------------------------------------------
1. Nhiệm vụ AGV
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW dbo.vw_rag_agv_tasks
AS
SELECT
    task_id,
    task_code,
    robot_code,
    task_status,
    source_location,
    target_location,
    error_code,
    error_message,
    created_at,
    completed_at
FROM dbo.YourAgvTaskTable;
GO
*/

/*
-------------------------------------------------------------------------------
2. Pallet theo vị trí
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW dbo.vw_rag_pallet_locations
AS
SELECT
    location_code,
    pallet_code,
    material_code,
    quantity,
    stock_status,
    updated_at
FROM dbo.YourPalletLocationTable;
GO
*/

/*
-------------------------------------------------------------------------------
3. Cuộn vải theo mã QR
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW dbo.vw_rag_fabric_rolls
AS
SELECT
    qr_code,
    material_code,
    fabric_name,
    pallet_code,
    location_code,
    stock_status,
    updated_at
FROM dbo.YourFabricRollTable;
GO
*/

/*
-------------------------------------------------------------------------------
4. Tài khoản đọc dành riêng cho RAG
-------------------------------------------------------------------------------

Ví dụ tổng quát. Hãy thực hiện theo chính sách tài khoản của công ty.
Không sử dụng tài khoản sysadmin hoặc tài khoản ứng dụng WMS chính.

CREATE LOGIN rag_reader WITH PASSWORD = 'REPLACE_WITH_STRONG_PASSWORD';
GO

USE YourDatabase;
GO

CREATE USER rag_reader FOR LOGIN rag_reader;
GO

GRANT SELECT ON dbo.vw_rag_agv_tasks TO rag_reader;
GRANT SELECT ON dbo.vw_rag_pallet_locations TO rag_reader;
GRANT SELECT ON dbo.vw_rag_fabric_rolls TO rag_reader;
GO

Không cấp INSERT, UPDATE, DELETE, EXECUTE, ALTER hoặc CONTROL.
*/
