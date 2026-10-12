
import argparse
import os
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Tham số dòng lệnh
# ============================================================

parser = argparse.ArgumentParser(
    description="Fakeddit image downloader"
)

parser.add_argument(
    "type",
    type=str,
    help="Path to train, validate, or test TSV file"
)

args = parser.parse_args()


# ============================================================
# Đọc TSV
# ============================================================

print(f"Loading dataset: {args.type}")

df = pd.read_csv(args.type, sep="\t")

# Thay giá trị NaN
df = df.replace(np.nan, "", regex=True)
df.fillna("", inplace=True)

print(f"Total samples: {len(df):,}")


# ============================================================
# Tạo thư mục ảnh
# ============================================================

IMAGE_DIR = "data/raw/fakeddit/images"

os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# Log ảnh tải thất bại
# ============================================================

FAILED_FILE = "data/raw/fakeddit/failed_images.txt"


# ============================================================
# Thiết lập tải ảnh
# ============================================================

MAX_RETRIES = 3
RETRY_DELAY = 3
REQUEST_DELAY = 0.3


# ============================================================
# Hàm tải ảnh
# ============================================================

def download_image(image_url, output_path):
    """
    Tải một ảnh và thử lại khi lỗi.

    Trả True nếu thành công, False nếu thất bại.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            # User-Agent giúp một số máy chủ chấp nhận yêu cầu
            request = urllib.request.Request(
                image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                }
            )

            with urllib.request.urlopen(
                request,
                timeout=20
            ) as response:

                with open(output_path, "wb") as f:
                    f.write(response.read())

            return True

        except urllib.error.HTTPError as e:

            # 429: quá nhiều yêu cầu
            if e.code == 429:

                wait_time = RETRY_DELAY * attempt

                print(
                    f"\nHTTP 429 for {image_url}"
                    f" | retry {attempt}/{MAX_RETRIES}"
                    f" | waiting {wait_time}s"
                )

                time.sleep(wait_time)

            else:

                print(
                    f"\nHTTP {e.code} for {image_url}"
                )

                return False

        except Exception as e:

            print(
                f"\nError downloading {image_url}: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return False

    return False


# ============================================================
# Tải ảnh
# ============================================================

success_count = 0
skip_count = 0
failed_count = 0


with open(FAILED_FILE, "a", encoding="utf-8") as failed_log:

    pbar = tqdm(
        total=len(df),
        desc="Downloading images"
    )

    for index, row in df.iterrows():

        image_id = str(row["id"]).strip()
        image_url = str(row["image_url"]).strip()

        # ----------------------------------------------------
        # Kiểm tra ảnh đã có
        # ----------------------------------------------------

        output_path = os.path.join(
            IMAGE_DIR,
            image_id + ".jpg"
        )

        if os.path.exists(output_path):

            skip_count += 1
            pbar.update(1)
            continue


        # ----------------------------------------------------
        # Kiểm tra khả năng tải ảnh
        # ----------------------------------------------------

        has_image = row["hasImage"]

        # Xử lý trường hợp giá trị là chuỗi
        if isinstance(has_image, str):

            has_image = has_image.lower() == "true"


        if not has_image:
            pbar.update(1)
            continue


        if image_url == "" or image_url.lower() == "nan":

            pbar.update(1)
            continue


        # ----------------------------------------------------
        # Tải ảnh
        # ----------------------------------------------------

        success = download_image(
            image_url,
            output_path
        )


        if success:

            success_count += 1

        else:

            failed_count += 1

            failed_log.write(
                f"{image_id}\t{image_url}\n"
            )

            failed_log.flush()


        # ----------------------------------------------------
        # Nghỉ ngắn giữa các yêu cầu
        # ----------------------------------------------------

        time.sleep(REQUEST_DELAY)

        pbar.update(1)


    pbar.close()


# ============================================================
# Tổng kết
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Total samples : {len(df):,}")
print(f"Downloaded    : {success_count:,}")
print(f"Skipped       : {skip_count:,}")
print(f"Failed        : {failed_count:,}")

print("=" * 60)

if failed_count > 0:
    print(f"Failed URLs saved to: {FAILED_FILE}")

print("done")

