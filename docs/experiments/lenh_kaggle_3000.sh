
# standard: fakeddit_2way (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_2way --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_2way --model dfn5b_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_2way --model siglip2_so400m --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_2way --model openai_clip_vit_b_32 --ensemble false --use-kaggle true

# standard: fakeddit_6way (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_6way --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_6way --model dfn5b_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_6way --model siglip2_so400m --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job fakeddit_6way --model openai_clip_vit_b_32 --ensemble false --use-kaggle true

# standard: mmimdb_genres (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job mmimdb_genres --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job mmimdb_genres --model dfn5b_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job mmimdb_genres --model siglip2_so400m --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job mmimdb_genres --model openai_clip_vit_b_32 --ensemble false --use-kaggle true

# standard: crisismmd_info (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_info --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_info --model dfn5b_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_info --model siglip2_so400m --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_info --model openai_clip_vit_b_32 --ensemble false --use-kaggle true

# standard: crisismmd_humanitarian (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_humanitarian --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_humanitarian --model dfn5b_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_humanitarian --model siglip2_so400m --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job crisismmd_humanitarian --model openai_clip_vit_b_32 --ensemble false --use-kaggle true

# adapted: adapted_crisismmd_info (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model metaclip2_vith14 --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model dfn5b_vith14 --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model siglip2_so400m --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model openai_clip_vit_b_32 --method dclip --use-kaggle true

# adapted: adapted_crisismmd_humanitarian (mặc định 3000 mẫu)
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model metaclip2_vith14 --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model dfn5b_vith14 --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model siglip2_so400m --method dclip --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model openai_clip_vit_b_32 --method dclip --use-kaggle true

# Tổng cộng: 28 lệnh độc lập. Chỉ chạy một lệnh mỗi lần.
