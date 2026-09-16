# Pose Teachable Lab

Ứng dụng học máy tương tác theo phong cách Teachable Machine dành cho Pose.

## Quy trình học sinh

1. Bật camera.
2. Tạo hoặc đổi tên class.
3. Giữ nút `HOLD TO RECORD` để thu dữ liệu tư thế.
4. Thu tối thiểu 5 mẫu cho mỗi class; khuyến nghị 30–60 mẫu/class.
5. Nhấn `TRAIN MODEL`.
6. Kiểm thử bằng tư thế mới và xem Confidence.
7. Bổ sung dữ liệu và Train lại nếu cần.

MoveNet chỉ dùng để lấy 17 điểm cơ thể. Bộ phân loại phía sau được tạo và huấn luyện trực tiếp trong trình duyệt bằng dữ liệu do học sinh thu.

## GitHub Pages

Website entry point: `index.html`.

Sau khi bật GitHub Pages từ nhánh `main` thư mục `/root`, website sẽ có dạng:

`https://nguyenminhsang7265-debug.github.io/PoseProject/`
