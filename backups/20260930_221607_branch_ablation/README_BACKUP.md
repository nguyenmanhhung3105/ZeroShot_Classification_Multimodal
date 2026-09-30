# Backup trước khi triển khai ablation

Đã copy và đối chiếu SHA-256 trước khi sửa:

- `src/inference.py`: `74642146befaf14193e968991ced477694a581557911719a75680287cf7adf70`
- `README.md`: `9d6b6a7717ce979e8dd6232b64f3850e6ba2c610d60e9b74e650708ade05217b`

Bản README sao lưu bao gồm các thay đổi chưa commit có trước đợt này.
Không sao lưu dataset, results, secrets, môi trường hoặc model weights.

File mới của đợt này (không tồn tại trước đó):

- `src/branch_ablation.py`
- `configs/ablation_config.yaml`
- `configs/branch_prompt_routes.json`
- `tests/test_branch_ablation.py`

Không tự động restore toàn bộ worktree: giữ các thay đổi khác của người dùng.
