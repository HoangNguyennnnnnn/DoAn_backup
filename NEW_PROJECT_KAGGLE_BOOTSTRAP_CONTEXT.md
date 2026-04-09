NEW PROJECT KICKOFF CONTEXT
Tên dự án gợi ý: MeshLatent-Kaggle
Ngày bắt đầu: 2026-04-07
Mục tiêu chính: Xây dựng dự án sinh 3D dạng mesh/voxel mới, tối ưu cho Kaggle runtime, dùng dataset 3D có sẵn trên Kaggle để tránh upload dữ liệu lớn và tận dụng GPU cloud.

1) Bài toán và mục tiêu sản phẩm
- Xây dựng pipeline 2 stage tương tự tư duy FaceDiff nhưng độc lập hoàn toàn:
  - Stage 1: Học latent geometry bằng autoencoder trên dữ liệu 3D mesh.
  - Stage 2: Học mô hình sinh latent (diffusion hoặc mean-flow) từ noise.
- Ưu tiên tính thực dụng:
  - Chạy ổn định trên Kaggle P100 và T4x2.
  - Có checkpoint trung gian để resume khi phiên Kaggle hết thời gian.
- Không phụ thuộc face-specific encoder trong phiên bản backup đầu tiên.

2) Phạm vi kỹ thuật
- In-scope (phiên bản v1):
  - Dữ liệu nguồn từ Kaggle datasets có sẵn.
  - Chuyển đổi OFF sang OBJ nếu cần.
  - Context encoder dùng DINO-only thay vì ArcFace.
  - Train Stage 1 đầy đủ, Stage 2 ở mức smoke test nếu thời gian hạn chế.
- Out-of-scope (v1):
  - Tái tạo chất lượng face identity như pipeline face chuyên dụng.
  - TPU optimization hoàn chỉnh ngay từ đầu.

3) Dataset strategy
Dataset ưu tiên:
- ModelNet40 - Princeton 3D Object Dataset.
  - Ưu điểm: phổ biến, ổn định, train/test split rõ, phù hợp pretrain latent shape.
  - Nhược điểm: OFF format, cần converter.

Dataset fallback:
- ShapeNetPart.
  - Dùng cho ablation hoặc mở rộng.
  - Dữ liệu thiên về part labels và point-based task.

Quy tắc chọn dataset:
- Ưu tiên nguồn có provenance/citation rõ.
- Ưu tiên mesh data dùng được trực tiếp hoặc convert đơn giản.
- Với mirror có license mơ hồ: chỉ dùng nội bộ, không publish weights công khai.

4) Tư duy thay ArcFace
Vì dataset mới là object generic, ArcFace không phù hợp.
Áp dụng backend context mới:
- Mặc định: DINO dual-view.
  - Front render embedding + back render embedding.
  - Context dimension đề xuất: 768.
- Fallback: DINO single-view.
  - Context dimension: 384.
- Legacy (không dùng mặc định): ArcFace + FLAME + DINO.

Bảng backend context:
- legacy_face: 946
- dino_dual_view: 768
- dino_single_view: 384

5) Kiến trúc hệ thống dự án mới
A. Data layer
- Mesh scanner: đọc obj/off theo recursive walk.
- Converter: OFF -> OBJ.
- Cache layer: lưu tensor đã convert để giảm I/O.

B. Stage 1 (Latent Autoencoder)
- Input: mesh features geom6 trước.
- Output: latent tokens fixed length.
- Loss:
  - reconstruction loss.
  - optional KL nếu dùng VAE.
- Checkpoint policy:
  - interrupt
  - latest_step
  - best

C. Stage 2 (Latent Generator)
- Input: noise + context embedding.
- Output: latent tokens.
- Mục tiêu v1:
  - train được pipeline end-to-end ở mức smoke.
  - chưa đặt KPI chất lượng cao.

6) Cấu trúc thư mục đề xuất cho repo mới
meshlatent-kaggle/
- README.md
- PROJECT_SCOPE.md
- DATA_LICENSE.md
- configs/
  - hardware_p100.yaml
  - hardware_t4x2.yaml
  - hardware_tpuv5e8.yaml
  - data_modelnet40.yaml
  - train_stage1.yaml
  - train_stage2.yaml
- scripts/
  - kaggle_bootstrap.sh
  - prepare_modelnet40.sh
  - train_stage1_autoresume.sh
  - train_stage2_autoresume.sh
  - export_artifacts.sh
- src/
  - data/
    - dataset_adapter.py
    - off_to_obj_converter.py
    - mesh_to_feature.py
  - models/
    - latent_autoencoder.py
    - latent_generator.py
    - context_backends/
      - dino_context.py
  - train/
    - train_stage1.py
    - train_stage2.py
  - inference/
    - generate_mesh.py
- notebooks/
  - kaggle_stage1_train.ipynb
  - kaggle_stage2_smoke.ipynb
- outputs/
- checkpoints/
- logs/

7) Kế hoạch theo phase
Phase 0 - Khởi tạo dự án (0.5-1 ngày)
- Tạo repo mới và skeleton thư mục.
- Viết README + quickstart.
- Tạo config phần cứng P100/T4x2.
Kết quả: repo chạy được bootstrap scripts.

Phase 1 - Data pipeline (1-2 ngày)
- Kết nối dataset input từ Kaggle.
- Convert OFF -> OBJ.
- Sinh cache feature.
- Tắt identity split filter mặc định.
Kết quả: dataloader trả về sample > 0 ổn định.

Phase 2 - Stage 1 training (2-4 ngày)
- Train autoencoder trên ModelNet40.
- Tích hợp resume + OOM backoff.
- Lưu checkpoints theo policy 3 mốc.
Kết quả: có best checkpoint dùng được.

Phase 3 - Stage 2 smoke (1-3 ngày)
- Dùng DINO context backend.
- Chạy smoke 1-3 epochs để xác minh pipeline.
Kết quả: train loop và save checkpoint hoạt động.

Phase 4 - Ổn định và đóng gói (1-2 ngày)
- Viết notebook run-all cho Kaggle.
- Đo benchmark P100 vs T4x2.
- Xuất artifact/checkpoint.
Kết quả: quy trình 1 lệnh cho người khác.

8) Runtime strategy theo phần cứng Kaggle
P100 profile (safe)
- Precision: fp16.
- Batch nhỏ, tăng grad accumulation.
- Num workers thấp để tránh nghẽn I/O.
- Save checkpoint thường xuyên.

T4x2 profile (balanced)
- Có thể DDP nếu code sẵn, nếu không thì single GPU mode vẫn ổn.
- Batch tăng vừa phải.
- Ưu tiên throughput cho stage 1.

TPU v5e-8 profile (future branch)
- Không phải mục tiêu khởi động v1 vì stack hiện tại thiên CUDA.
- Cần nhánh riêng với PyTorch/XLA hoặc JAX.
- Chỉ bắt đầu sau khi GPU pipeline ổn định.

9) KPI và định nghĩa hoàn thành
KPI kỹ thuật tối thiểu:
- Stage 1 chạy từ đầu đến latest_step checkpoint trên Kaggle.
- Resume thành công sau khi ngắt phiên.
- Throughput ổn định không crash kéo dài.

KPI chất lượng tối thiểu:
- Reconstruction loss giảm đều qua epochs.
- Inference decode cho mesh hợp lệ.

Definition of Done (v1 backup):
- Có notebook Kaggle stage 1 run-all.
- Có artifacts checkpoint best và latest.
- Có runbook tái lập trong README.
- Có context backend dino_dual_view hoạt động.

10) Rủi ro và phương án giảm thiểu
Rủi ro 1: Session Kaggle hết giờ.
- Giảm thiểu: checkpoint theo step, autoresume script.

Rủi ro 2: I/O chậm do convert/caching lớn.
- Giảm thiểu: cache warmup trước train, prefetch vừa phải.

Rủi ro 3: Domain gap object -> face.
- Giảm thiểu: xác định rõ mục tiêu backup là pretrain latent, không kỳ vọng face quality.

Rủi ro 4: TPU chưa tương thích.
- Giảm thiểu: chốt GPU-first trong v1.

11) Quy chuẩn thực thi
- Mỗi thay đổi lớn cần ghi vào WORK_LOG.
- Mỗi run training cần lưu:
  - config snapshot.
  - git commit hash (nếu có).
  - hardware profile.
  - checkpoint path.
- Không hard-code đường dẫn local.

12) Quy trình bắt đầu trong ngày đầu tiên
Ngày 1 checklist:
- B1: Tạo repo mới và thư mục theo cấu trúc đã chốt.
- B2: Thêm script bootstrap Kaggle environment.
- B3: Tích hợp converter OFF -> OBJ.
- B4: Chạy dataloader smoke.
- B5: Chạy stage 1 trong 200-500 steps để xác minh checkpoint.

13) Text để dùng làm phần mở đầu README của dự án mới
MeshLatent-Kaggle là dự án backup cho bài toán sinh 3D latent-first trên Kaggle. Dự án được thiết kế để không phụ thuộc upload dataset lớn từ local, tận dụng dataset có sẵn trên Kaggle và tối ưu cho runtime GPU giới hạn thời gian. Phiên bản đầu tập trung vào Stage 1 latent autoencoder với dataset ModelNet40, sử dụng DINO-based context cho khả năng mở rộng sang Stage 2 trong các lần lặp tiếp theo.

14) Tuyên bố kiến trúc ngắn gọn
- Data in Kaggle -> Mesh preprocessing -> Latent autoencoder -> Latent generator -> Mesh decode.
- Context backend mặc định: DINO dual-view.
- Hardware target v1: P100 và T4x2.
- TPU branch: triển khai sau khi pipeline GPU ổn định.

15) Quyết định kỹ thuật đã chốt cho dự án mới
- Không dùng ArcFace trong default path.
- Dùng DINO-based context.
- Ưu tiên train Stage 1 hoàn chỉnh trước Stage 2.
- Ưu tiên khả năng resume/recovery hơn tốc độ peak.
- Tách cấu hình theo hardware profile.

END OF KICKOFF CONTEXT
