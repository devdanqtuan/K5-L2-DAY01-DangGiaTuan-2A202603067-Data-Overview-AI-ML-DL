# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU / GPU (T4 GPU trên Google Colab)

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   **Trả lời:** Record này nói về **cả bức ảnh** (image-level classification). Lý do là vì Image Classification phân loại toàn bộ bối cảnh không gian của ảnh thành một phân phối xác suất trên danh mục lớp định sẵn, hoàn toàn không chứa thông tin tọa độ hình học (`bbox_xyxy` hay `polygon_xy`) để khoanh vùng riêng cho bất kỳ vật thể cụ thể nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   **Trả lời:** Danh sách tên lớp **đi kèm taxonomy và checkpoint của mô hình** (cụ thể ở đây là taxonomy ImageNet-1K gồm 1,000 danh mục lớp được chuẩn hóa sẵn kèm checkpoint `yolo11n-cls.pt`), hoàn toàn không phải do mô hình tự nghĩ ra. Mô hình chỉ tính toán trọng số xác suất dựa trên các lớp có sẵn trong taxonomy này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   **Trả lời:** Cần ghi đủ cả ba trường thông tin vì:
   - `class_id` là mã số định danh chuẩn giúp hệ thống máy tính xử lý, không bị lỗi chính tả.
   - `class_name` giúp con người (annotator, reviewer, kỹ sư) đọc hiểu và nghiệm thu trực quan, biết được cái đó nó là cái gì.
   - `taxonomy_name` xác định không gian ngữ nghĩa (ontology/namespace) đang sử dụng. Cùng một tên lớp (ví dụ: `cab`, `table`) có thể có ý nghĩa hoặc mã số hoàn toàn khác nhau giữa các bộ dữ liệu khác nhau (như ImageNet-1K so với COCO-80 hay OpenImages).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   **Trả lời:** Guideline cần đưa ra **quy tắc ưu tiên rõ ràng (priority & dominance rules)**:
   - Ưu tiên đối tượng nổi bật nhất (salient object) chiếm diện tích lớn nhất trong khung hình hoặc nằm ở vị trí trung tâm / tiền cảnh (foreground).
   - Quy định rõ thứ tự phân cấp chủ đề: ví dụ nếu là cảnh giao thông hỗn hợp (có cả taxi, xe buýt, người đi bộ, tòa nhà), guideline phải chỉ rõ chọn theo bối cảnh chung của ảnh hay chọn phương tiện chiếm điểm nhìn trọng tâm.
   - Hướng dẫn cách xử lý phân vân giữa các lớp gần nghĩa (ví dụ: `cab` vs `minibus` vs `passenger car`).
6. Vì sao không được lấy `score` của model làm ground truth?  
   **Trả lời:** Vì `score` (confidence) chỉ là độ tin cậy xác suất toán học của mô hình dựa trên các tham số trọng số hiện tại của mạng nơ-ron; nó phản ánh mức độ tự tin của thuật toán chứ không phải sự thật khách quan. Mô hình hoàn toàn có thể tự tin rất cao (score > 0.9) vào một dự đoán sai (overconfidence/hallucination) hoặc tự tin thấp vào một vật thể đúng do ảnh bị mờ. Ground truth bắt buộc phải là nhãn đúng do con người xác thực dựa theo guideline chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   **Trả lời:** Trong hệ tọa độ pixel của ảnh với gốc tọa độ `(0, 0)` tại góc trên cùng bên trái:
   - Góc trên - trái của bounding box nằm tại tọa độ pixel `(x_min = 385.33, y_min = 69.24)`.
   - Góc dưới - phải của bounding box nằm tại tọa độ pixel `(x_max = 498.92, y_max = 348.92)`.
   - Kích thước của hộp chữ nhật này có chiều rộng là `113.58` pixel và chiều cao là `279.68` pixel, bao bọc vừa vặn người phụ nữ đang đứng ở bên phải phòng bếp.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   **Trả lời:**
   - Ngưỡng `0.20`: có **17** predictions (gồm: 2 person, 5 bowl, 2 oven, 2 cup, 3 spoon, 1 potted plant, 1 dining table, 1 bottle).
   - Ngưỡng `0.60`: có **6** predictions (gồm: 2 person, 2 bowl, 2 oven).  
   - Ngưỡng mặc định `0.35`, có **11** predictions.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   **Trả lời:** Khi hạ ngưỡng (ví dụ từ 0.60 xuống 0.20), số lượng bounding box tăng mạnh (từ 6 lên 17 hộp) và kéo theo nhiều dự đoán có độ tin cậy thấp. Người reviewer sẽ phải:
   - Tốn thêm nhiều thời gian rà soát từng hộp để **loại bỏ các dự đoán dương tính giả (false positives)** do mô hình đoán nhầm nền hoặc vật thể không liên quan.
   - **Chỉnh sửa lại tọa độ viền hộp** bị lệch, thừa khoảng trống hoặc cắt phạm vào vật thể.
   - **Sửa lại phân loại lớp (class name)** nếu mô hình gán nhãn nhầm đối tượng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   **Trả lời:** Quy tắc: *"Bounding box phải ôm sát khít toàn bộ phần nhìn thấy được của vật thể, khoảng hở giữa biên hộp và đường viền ngoài cùng của vật thể không được vượt quá 2 pixel, và tuyệt đối không được cắt lẹm vào các chi tiết cấu thành của vật thể."*
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   **Trả lời:**
   - *Vật thể quan sát được:* Đối tượng người ở mép trái bức ảnh (`bbox_xyxy` khoảng `[0.12, 263.18, 61.55, 310.78]`) chỉ thấy một phần cánh tay bị cắt mép ảnh, hoặc các chiếc bát (`bowl`) xếp chồng lên nhau khiến phần thân dưới bị che khuất (occluded).
   - *Cách xử lý:* **Viết thành quy tắc trong guideline**.
   - *Vì sao:* Tình huống vật thể bị cắt mép hoặc che khuất một phần là hiện tượng rất phổ biến trong thị giác máy tính thực tế. Việc chuẩn hóa thành điều khoản rõ ràng trong guideline (ví dụ: quy định *"Chỉ gán nhãn vật thể nếu diện tích nhìn thấy được từ 20% trở lên; nếu dưới 20% thì bỏ qua"*) giúp toàn bộ annotator áp dụng đồng nhất trên quy mô hàng nghìn bức ảnh mà không cần tốn thời gian escalation làm tắc nghẽn công việc của Lab Coach. Chỉ thực hiện escalation khi gặp vật thể biến dạng dị thường hoặc khi guideline chưa có tiền lệ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   **Trả lời:** So với bounding box vốn chỉ là một khung chữ nhật thô bao hàm cả nhiều điểm ảnh nền (background clutter), polygon cung cấp:
   - **Đường biên thực tế chính xác ở cấp độ điểm ảnh (pixel-level contour):** bám theo hình dáng giải phẫu cơ thể người, đường cong cánh tay, nếp gấp trang phục.
   - **Thông tin hình học và diện tích thực (geometry & true area):** biết chính xác chu vi, diện tích thực của vật thể, các góc lồi lõm hoặc khoảng rỗng bên trong mà hộp chữ nhật không thể thể hiện.
   - **Tách biệt ranh giới giữa tiền cảnh (vật thể) và hậu cảnh:** cho phép mô hình học chính xác cấu trúc hình thể mà không bị nhiễu nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   **Trả lời:**
   - `instance_id` (ví dụ: `kitchen-001`) dùng để phân biệt độc lập từng cá thể đối tượng trong cùng một bức ảnh tĩnh, đảm bảo mỗi thực thể là một thực thể riêng biệt dù có cùng tên lớp.
   - *Không phải class ID:* Vì class ID là mã số định danh phân loại chung của một nhóm loài/loại vật thể trong taxonomy (ví dụ: trong COCO-80, tất cả người đều có `class_id = 0`). Còn `instance_id` phân biệt cụ thể "người thứ nhất" với "người thứ hai".
   - *Không phải tracking ID:* Vì tracking ID là mã định danh xuyên suốt chuỗi thời gian qua các frame của video để theo vết hành trình di chuyển của cùng một đối tượng. `instance_id` chỉ có giá trị cục bộ trong phạm vi một ảnh tĩnh duy nhất của bài lab.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   **Trả lời:** Quy tắc: *"Khi vẽ đường viền đa giác (polygon mask), các điểm định vị phải đi khít theo ranh giới quang học ngoài cùng của vật thể; sai số lệch nền không vượt quá 2 pixel; tại các góc cong mượt phải đặt mật độ điểm đều đặn (khoảng 5-10 pixel/điểm) để bảo toàn độ cong tự nhiên, không được vẽ đường thẳng thô cắt xén hình dạng vật thể."*
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   **Trả lời:**
   - *Chỗ quan sát được:* Vùng tiếp giáp giữa các chiếc bát (`bowl`) đặt sát nhau trên kệ/bàn, hoặc phần mép bàn bếp tiếp giáp với tường gạch bóng có phản chiếu ánh sáng gây nhòe biên.
   - *Cách xử lý:* **Dùng guideline**.
   - *Vì sao:* Đây là đặc trưng quang học phổ biến trong ảnh chụp nội thất. Guideline cần quy định sẵn nguyên tắc xử lý va chạm ranh giới: *"Khi hai vật thể chạm nhau, đường biên polygon của vật thể phía trước (tiền cảnh) được vẽ trọn vẹn theo rìa quang học nhìn thấy; vật thể phía sau vẽ bám theo mép che chắn, tuyệt đối không tạo vùng mặt nạ chồng lấn (no overlapping mask)."* Việc này giúp định hướng rõ ràng cho annotator thực hiện tự động và nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp danh mục duy nhất cho toàn bộ bức ảnh (image-level label: `class_id`, `class_name`). | Cảnh `traffic` có nhiều loại phương tiện hỗn hợp; model đoán rank 1 là `cab` (score: 0.5109) và rank 2 là `minibus` (0.1643) gây thiếu bao quát toàn cảnh giao thông. | Đối chiếu ảnh với quy tắc chủ đề nổi bật/bao trùm trong guideline; chọn đúng 1 nhãn lớp trong danh mục taxonomy chuẩn. | Kiểm tra nhãn được gán có đúng quy tắc ưu tiên không; phát hiện các ca gán nhãn chủ quan hoặc sai taxonomy. |
| Phát hiện vật thể | Tập hợp các hộp chữ nhật bao quanh từng vật thể (`bbox_xyxy` theo pixel) kèm `class_id`, `class_name`. | Trên ảnh `kitchen`, vật thể nhỏ/bị che như `spoon`, `cup` hoặc vật ở rìa trái (`bbox=[0.12, 263.18, 61.55, 310.78]`) dễ bị sót; khi nâng threshold lên 0.60 số hộp rớt từ 17 xuống 6. | Xác định từng vật thể thấy được thuộc 80 lớp COCO; đóng khung chữ nhật khít sát (tight box); gắn cờ nếu bị che khuất/cắt mép. | Soi kỹ độ ôm sát của viền box (không thừa viền, không phạm vật); rà soát sót vật thể (False Negative) và nhầm lớp. |
| Instance segmentation | Tập hợp các đa giác kín (`polygon_xy` gồm danh sách tọa độ pixel) cho từng cá thể riêng biệt kèm `instance_id`, `class_name`. | Đường viền quanh người phụ nữ `kitchen-001` rất phức tạp (348 điểm); các miệng bát, đĩa chạm nhau trên bàn dễ bị vẽ chồng lấn hoặc răng cưa. | Dùng công cụ polygon vẽ bám sát viền thực của từng cá thể; tách riêng từng vật dù cùng lớp `bowl` hay `person`; không để mặt nạ đè nhau. | Phóng to kiểm tra độ mượt và chính xác của viền mask ở cấp độ pixel; đảm bảo không hở viền, không chồng lấn; kiểm tra `instance_id` duy nhất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   **Trả lời:** *"Chỉ sử dụng các tệp ảnh COCO công khai được ghim mã định danh và kiểm tra mã băm SHA-256 trong bài lab; tuyệt đối không tải lên, lưu trữ hoặc công khai hình ảnh cá nhân, khuôn mặt, biển số xe, dữ liệu khách hàng hoặc dữ liệu bảo mật nội bộ của VinFast / VinUni."*
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   **Trả lời:** *"Tôi sẽ lập tức dừng thực hiện, không lưu trữ hoặc xử lý thêm bất kỳ thao tác nào đối với dữ liệu đó, và báo cáo ngay lập tức cho Giảng viên / Lab Coach phụ trách lớp học để được hướng dẫn xử lý cách ly an toàn."*

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
