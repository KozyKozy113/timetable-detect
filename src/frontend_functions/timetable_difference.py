from PIL import Image, ImageChops

def output_difference(image1, image2):
    # 両方の画像を比較するために横幅を揃える
    target_width = min(image1.width, image2.width)  # 小さい方の幅に合わせる
    new_height1 = int(image1.height * (target_width / image1.width))  # アスペクト比を保持
    new_height2 = int(image2.height * (target_width / image2.width))
    # サイズ変更（アスペクト比を維持）
    resized_image1 = image1.resize((target_width, new_height1), Image.LANCZOS)
    resized_image2 = image2.resize((target_width, new_height2), Image.LANCZOS)
    # 高さを一致させるためにパディング
    final_height = max(new_height1, new_height2)  # 高さを大きい方に揃える
    # 上下にパディングを追加
    padded_image1 = Image.new("RGB", (target_width, final_height), color=(0, 0, 0))
    padded_image1.paste(resized_image1, (0, (final_height - new_height1) // 2))
    padded_image2 = Image.new("RGB", (target_width, final_height), color=(0, 0, 0))
    padded_image2.paste(resized_image2, (0, (final_height - new_height2) // 2))
    # 画像の差分を計算
    difference = ImageChops.difference(padded_image1, padded_image2)
    return difference