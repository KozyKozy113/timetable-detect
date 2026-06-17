from PIL import Image, ImageChops
import numpy as np
import cv2

# 1ピクセル単位で「変化あり」とみなす差分強度の下限（リサイズ/圧縮由来の微小ノイズを除外）
PIXEL_NOISE_FLOOR = 30

# 差分矩形検出のパラメータ
# 横長カーネル: 文字の細かい差分を「語・行」単位の塊に結合する（文字変更対策の要）
DIFF_REGION_KERNEL = (25, 7)
# 1つの矩形が含むべき「実際に変化したピクセル」数の下限（これ未満はノイズとして除外）
# ※膨張後の面積ではなく膨張前の差分ピクセル数で判定するため、点ノイズに強い
DIFF_REGION_MIN_CHANGED = 30


def align_image_pair(image1, image2):
    """2画像をアスペクト比保持で同サイズに揃える。

    小さい方の幅に合わせて両方をリサイズし、高さは大きい方に合わせて
    上下中央パディングする。差分画像表示用（座標系の保持は不要なケース）。

    Returns:
        (padded_image1, padded_image2)
    """
    target_width = min(image1.width, image2.width)  # 小さい方の幅に合わせる
    new_height1 = int(image1.height * (target_width / image1.width))  # アスペクト比を保持
    new_height2 = int(image2.height * (target_width / image2.width))
    resized_image1 = image1.resize((target_width, new_height1), Image.LANCZOS)
    resized_image2 = image2.resize((target_width, new_height2), Image.LANCZOS)
    final_height = max(new_height1, new_height2)  # 高さを大きい方に揃える
    padded_image1 = Image.new("RGB", (target_width, final_height), color=(0, 0, 0))
    padded_image1.paste(resized_image1, (0, (final_height - new_height1) // 2))
    padded_image2 = Image.new("RGB", (target_width, final_height), color=(0, 0, 0))
    padded_image2.paste(resized_image2, (0, (final_height - new_height2) // 2))
    return padded_image1, padded_image2


def output_difference(image1, image2):
    """2画像の差分画像を返す（従来互換）。"""
    padded_image1, padded_image2 = align_image_pair(image1, image2)
    return ImageChops.difference(padded_image1, padded_image2)


def changed_area_ratio(diff_crop, noise_floor=PIXEL_NOISE_FLOOR):
    """差分画像領域における「変化したピクセル」の割合を 0-100(%) で返す。

    各ピクセルの RGB 最大チャンネル差分が noise_floor を超えたものを変化とみなす。
    局所的な変化（テキスト1行の差し替え等）を平均値より敏感に拾える。
    """
    if diff_crop.width == 0 or diff_crop.height == 0:
        return 0.0
    arr = np.asarray(diff_crop.convert("RGB"), dtype=np.uint8)
    per_pixel_max = arr.max(axis=2)  # 各ピクセルの最大チャンネル差分
    changed = np.count_nonzero(per_pixel_max > noise_floor)
    total = per_pixel_max.size
    return changed / total * 100.0


def detect_diff_regions(old_crop, new_crop, noise_floor=PIXEL_NOISE_FLOOR,
                        kernel_size=DIFF_REGION_KERNEL,
                        min_changed=DIFF_REGION_MIN_CHANGED):
    """新旧ステージ領域から差分のある矩形領域を検出する。

    文字変更のように差分が細かく散らばるケースでも、横長カーネルの
    モルフォロジー処理で「語・行」単位の塊にまとめてから連結成分を抽出する。
    各塊は「膨張前の実差分ピクセル数」で足切りするため、点ノイズが膨張で
    肥大化しても誤検出しない。

    Args:
        old_crop, new_crop: 同サイズの PIL.Image（ステージ領域）
    Returns:
        [(x, y, w, h), ...]  crop座標系の矩形リスト
    """
    old_arr = np.asarray(old_crop.convert("L"), dtype=np.int16)
    new_arr = np.asarray(new_crop.convert("L"), dtype=np.int16)
    if old_arr.shape != new_arr.shape or old_arr.size == 0:
        return []

    diff = np.abs(old_arr - new_arr).astype(np.uint8)
    raw_mask = (diff > noise_floor)  # 膨張前の実差分マスク
    grown = (raw_mask.astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # クローズで途切れを埋め、膨張で近接する文字差分を1つの塊に結合
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, kernel)
    grown = cv2.dilate(grown, kernel, iterations=1)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(grown, connectivity=8)
    regions = []
    for i in range(1, num):  # 0 は背景
        x, y, w, h, _area = stats[i]
        # 膨張後の塊(=ラベルi)に含まれる「実差分ピクセル」の数で足切り
        changed = int(np.count_nonzero(raw_mask[labels == i]))
        if changed < min_changed:
            continue
        regions.append((int(x), int(y), int(w), int(h)))
    # 上から順（読み順）に並べる
    regions.sort(key=lambda r: (r[1], r[0]))
    return regions


def draw_diff_overlay(image, regions, color=(255, 0, 0),
                      hatch_spacing=10, alpha=0.35, frame_thickness=2):
    """ステージ領域画像に、差分矩形の枠線と斜線ハッチングを重ねた画像を返す。

    Args:
        image: ベースとなる PIL.Image（ステージ領域）
        regions: detect_diff_regions が返す [(x,y,w,h), ...]
    Returns:
        PIL.Image（RGB）
    """
    base = np.array(image.convert("RGB"))
    h_img, w_img = base.shape[:2]
    if not regions:
        return Image.fromarray(base)

    # 全面に45度の斜線を引いたハッチ層を作る
    hatch = np.zeros_like(base)
    for c in range(-h_img, w_img, hatch_spacing):
        cv2.line(hatch, (c, 0), (c + h_img, h_img), color, 1)

    # 矩形内のみハッチを適用する領域マスク
    region_mask = np.zeros((h_img, w_img), dtype=bool)
    for (x, y, w, h) in regions:
        region_mask[y:y + h, x:x + w] = True
    hatch_mask = (hatch.sum(axis=2) > 0) & region_mask

    out = base.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[hatch_mask] = (base[hatch_mask] * (1 - alpha) + color_arr * alpha).astype(np.uint8)

    # 矩形枠（ハッチより前面にはっきり描画）
    for (x, y, w, h) in regions:
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, frame_thickness)
    return Image.fromarray(out)


def analyze_difference_by_stage(new_image, old_image, stage_list):
    """旧画像の座標系(bbox)で新旧を比較し、全体差分とステージ別スコアを返す。

    bbox は旧画像(raw.png)の座標系で定義されているため、新画像を旧画像と同じ
    サイズにリサイズして座標系を揃えてから比較する。

    Args:
        new_image: アップロードされた新画像 (PIL.Image または file-like)
        old_image: 既存画像 raw.png (PIL.Image)
        stage_list: project_info_json の timetables[].stage_list
                    （各要素に stage_no / stage_name / bbox{left,top,right,bottom}）

    Returns:
        dict:
            old_image:  RGB に変換した既存画像
            new_image:  旧画像サイズに揃えた新画像 (RGB)
            diff_image: 全体差分画像
            stages: [{stage_no, stage_name, score, diff_crop, old_crop, new_crop,
                      regions, old_overlay, new_overlay}, ...]
                    score は変化ピクセル割合(0-100%)、*_crop は各ステージ領域画像、
                    regions は差分矩形 [(x,y,w,h),...]、*_overlay は矩形+ハッチ重畳画像
    """
    if not isinstance(old_image, Image.Image):
        old_image = Image.open(old_image)
    if not isinstance(new_image, Image.Image):
        new_image = Image.open(new_image)

    old_rgb = old_image.convert("RGB")
    new_rgb = new_image.convert("RGB").resize(old_rgb.size, Image.LANCZOS)
    diff_image = ImageChops.difference(old_rgb, new_rgb)

    stages = []
    for stage in stage_list or []:
        bbox = stage.get("bbox")
        if bbox is None:
            continue
        box = (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"])
        diff_crop = diff_image.crop(box)
        old_crop = old_rgb.crop(box)
        new_crop = new_rgb.crop(box)
        regions = detect_diff_regions(old_crop, new_crop)
        stages.append({
            "stage_no": stage.get("stage_no"),
            "stage_name": stage.get("stage_name", ""),
            "score": changed_area_ratio(diff_crop),
            "diff_crop": diff_crop,
            "old_crop": old_crop,
            "new_crop": new_crop,
            "regions": regions,
            "old_overlay": draw_diff_overlay(old_crop, regions),
            "new_overlay": draw_diff_overlay(new_crop, regions),
        })

    return {
        "old_image": old_rgb,
        "new_image": new_rgb,
        "diff_image": diff_image,
        "stages": stages,
    }
