import math
import os
from PIL import Image, ImageDraw, ImageFont
import datetime

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.abspath(os.path.join(DIR_PATH, "Fonts/NotoSansJP-Regular.otf"))

# 可読性目標
MIN_FONT_SIZE = 12               # 縮小許容下限
MAX_NAME_LINES = 2               # アーティスト名の許容行数


def _probe_line_height_ratio() -> float:
    """フォントの実 line height / font_size 比を測定する（NotoSansJP ≈ 1.37）。"""
    try:
        probe = ImageFont.truetype(font_path, 1000)
        ascent, descent = probe.getmetrics()
        return (ascent + descent) / 1000.0
    except Exception:
        return 1.4


# 実フォントの ascent + descent から算出した行高比（描画時の真の line height）
LINE_HEIGHT_RATIO = _probe_line_height_ratio()

# 標準縦横比（高さ / 幅）
# 2 行表示が破綻しない最低ライン。これより枠が横長になる場合は 1 行表示に切替。
# 1:5 を基準とする（1:3 だと細長判定が緩すぎ、font が暴走しやすかったため）
TARGET_BOX_ASPECT_2LINE = 1.0 / 5.0

# 単体可読性のためのデフォルト PPM（time_line_spacing 指定なしフォールバック用）
TARGET_PPM = 5.0

# 生成画像の絶対サイズ上限（メモリ・描画コスト保護）
MAX_GEN_HEIGHT = 4000
MAX_GEN_WIDTH = 2000

# レイアウトの内部定数
_MARGIN = 10                     # 周囲のマージン
_BOX_MARGIN = 20                 # 出演枠の四角の左右マージン
_TEXT_MARGIN = 10                # 四角内のテキスト表示マージン
_DUPLICATE_MARGIN_RATIO = 0.05   # 出演枠の必要内側幅に対する重複インデント比率
_DUPLICATE_MARGIN_MIN = 8        # 重複インデントの最小値 (px)
_TIMELINE_TEXT_MARGIN = 60       # 時間軸（値）表示幅
_BOX_VPAD = 2                    # box 内の上下余白（テキスト上下に確保）
_ONELINE_SPACER = "　 "          # 1 行モードでのグループ名と時間の区切り（全角＋半角空白）
_WIDTH_SAFETY_PAD = 2            # サブピクセル/アンチエイリアス用の安全マージン (px)


def _wrap_by_pixel(text: str, font: ImageFont.FreeTypeFont, max_px: float) -> list:
    """1 文字ずつ実測しながら greedy に折り返す。日英混在に対応。"""
    lines = []
    cur = ""
    for ch in text:
        if font.getlength(cur + ch) <= max_px or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _truncate_lines(lines: list, font: ImageFont.FreeTypeFont, max_px: float, max_lines: int) -> list:
    """max_lines を超えたら末尾を … で省略。"""
    if len(lines) <= max_lines:
        return lines
    result = list(lines[:max_lines])
    last = result[-1]
    while last and font.getlength(last + "…") > max_px:
        last = last[:-1]
    result[-1] = last + "…"
    return result


def _diff_minutes(from_str: str, to_str: str, time_format: str = "%H:%M") -> int:
    """time_text 表示用の (分) 値を再計算する。"""
    try:
        f = datetime.datetime.strptime(from_str, time_format)
        t = datetime.datetime.strptime(to_str, time_format)
        return max(int((t - f).total_seconds() / 60), 1)
    except Exception:
        return 0


def _build_tokutenkai_view_json(json_data: dict) -> dict:
    """特典会併記JSON → 特典会列描画用の擬似ライブJSON。

    各 live の 特典会[] の各要素を擬似ライブ枠に変換。
    複数特典会のときライブ枠を時間軸で N 等分し、末尾は live to に合わせる。
    擬似 live の ライブステージ.from/to は描画の Y 位置のための仮想時刻。
    time_text 表示には _display_time_from / _display_time_to を別途持たせる。
    """
    time_format = "%H:%M"
    new_timetable = []

    for live in json_data.get("タイムテーブル", []):
        tklist_raw = live.get("特典会", []) or []
        # ブース名 / from / to のいずれかが欠けている要素はスキップ（描画対象外）
        tklist = [
            tk for tk in tklist_raw
            if (tk.get("ブース") or "").strip()
            and (tk.get("from") or "").strip()
            and (tk.get("to") or "").strip()
        ]
        if not tklist:
            continue  # 特典会なし → 右側空白

        try:
            live_from = datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
            live_to = datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
        except (KeyError, ValueError, TypeError):
            continue

        total_min = int((live_to - live_from).total_seconds() / 60)
        n = len(tklist)
        # 分単位の境界（末尾は live_to と一致）
        boundary = [round(total_min * i / n) for i in range(n + 1)]
        boundary[-1] = total_min

        for i, tk in enumerate(tklist):
            sub_from = live_from + datetime.timedelta(minutes=boundary[i])
            sub_to = live_from + datetime.timedelta(minutes=boundary[i + 1])
            # 退化（sub_from == sub_to）を防ぐため最低 1 分
            if sub_to <= sub_from:
                sub_to = sub_from + datetime.timedelta(minutes=1)

            booth = tk["ブース"].strip()

            new_timetable.append({
                "グループ名": booth,
                "グループ名_採用": booth,
                "ライブステージ": {
                    "from": sub_from.strftime(time_format),
                    "to": sub_to.strftime(time_format),
                },
                "_display_time_from": tk["from"],
                "_display_time_to": tk["to"],
            })

    return {
        "ステージ名": json_data.get("ステージ名", ""),
        "タイムテーブル": new_timetable,
    }


def _hstack_images(left: Image.Image, right: Image.Image) -> Image.Image:
    """左右に画像を横並びで合体。高さが違う場合は右の画像を左の高さに合わせてリサイズ。"""
    h = left.height
    if right.height != h:
        new_w = max(1, round(right.width * h / right.height))
        right = right.resize((new_w, h), Image.LANCZOS)
    combined = Image.new("RGB", (left.width + right.width, h), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    return combined


def create_timetable_image(
    json_data,
    start_margin=None,
    time_line_spacing=None,
    box_color="yellow",
    image_height=None,
    source_box_width=None,
    show_timeline_labels=True,
    apply_max_width_clamp=True,
):
    """タイムテーブル画像を生成する。

    横幅は **各出演枠の必要横幅から動的に決定** する。
    呼び出し側は縦軸 (start_margin / time_line_spacing / image_height) と
    元画像の box 横幅 (source_box_width) のみ渡す。
    """
    if "タイムテーブル" not in json_data.keys() or len(json_data["タイムテーブル"]) == 0:
        return None
    time_format = "%H:%M"
    json_data_timetable = []
    for live in json_data["タイムテーブル"]:
        try:
            datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
            datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
            json_data_timetable.append(live)
        except Exception:
            continue
    json_data["タイムテーブル"] = json_data_timetable
    if len(json_data["タイムテーブル"]) == 0:
        return None

    min_event_min = min(
        int((datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
             - datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)).total_seconds() / 60)
        for live in json_data["タイムテーブル"]
    )
    min_event_min = max(min_event_min, 1)

    # タイムテーブルの時間範囲を計算（30分刻みに拡張）
    raw_start = min(datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
                    for live in json_data["タイムテーブル"])
    raw_end = max(datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
                  for live in json_data["タイムテーブル"])
    start_time = raw_start.replace(minute=0)
    end_time = raw_end.replace(minute=0) + datetime.timedelta(hours=1)
    total_minutes = int((end_time - start_time).total_seconds() / 60)

    # 内部レイアウト定数（timeline_text_margin / duplicate_margin は後段で動的決定）
    margin = _MARGIN
    box_margin = _BOX_MARGIN
    text_margin = _TEXT_MARGIN

    # --- ステップ A: 縦軸パラメータの確定 ---
    if time_line_spacing is None:
        time_line_spacing = TARGET_PPM * 30
    ppm = time_line_spacing / 30.0
    if start_margin is None:
        start_margin = 30
    start_margin = int(start_margin)
    if image_height is None:
        image_height = int(start_margin + margin + (total_minutes // 30) * time_line_spacing)
    image_height = int(image_height)

    # --- ステップ B: レイアウト形式（1行 or 2行）判定 ---
    # 元画像の box 縦横比 = (ppm × 最短イベント) / 元画像 box 横幅
    box_height_min = ppm * min_event_min
    if source_box_width is not None and source_box_width > 0:
        source_aspect_min = box_height_min / source_box_width
        one_line_mode = source_aspect_min < TARGET_BOX_ASPECT_2LINE
    else:
        # フォールバック: 元画像情報がなければ 2 行モードを基本に
        one_line_mode = False

    # --- ステップ C: フォントサイズ決定（最小イベント基準・余白考慮） ---
    base_line_count = 1 if one_line_mode else 2
    avail_h_min = max(1.0, box_height_min - 2 * _BOX_VPAD)
    font_size_from_height = avail_h_min / (base_line_count * LINE_HEIGHT_RATIO)
    text_font_size = max(MIN_FONT_SIZE, int(font_size_from_height))

    try:
        if not os.path.exists(font_path):
            print("フォントファイルが見つかりません:", font_path)
        font = ImageFont.truetype(font_path, text_font_size)
    except IOError:
        print("font error")
        font = ImageFont.load_default()
    # 実フォントの ascent + descent を line height に採用（描画と完全一致させる）
    try:
        _ascent, _descent = font.getmetrics()
        line_height = max(1, _ascent + _descent)
    except Exception:
        line_height = max(1, int(round(text_font_size * LINE_HEIGHT_RATIO)))
    spacer_width = font.getlength(_ONELINE_SPACER)

    # 時間軸ラベル幅を実測し、固定値 _TIMELINE_TEXT_MARGIN と比較して大きい方を採用
    if show_timeline_labels:
        time_label_width = font.getlength("00:00")
        timeline_text_margin = max(
            _TIMELINE_TEXT_MARGIN,
            int(time_label_width) + text_margin,
        )
    else:
        timeline_text_margin = 0

    # --- ステップ D: 各イベントの折り返し判定 & 必要横幅算出 ---
    # 必要内側幅にはサブピクセル安全マージンを上乗せする
    target_box_width_scaled = source_box_width if (source_box_width is not None and source_box_width > 0) else None

    per_event = []
    for live in json_data["タイムテーブル"]:
        start = datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
        end = datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
        minutes = max(int((end - start).total_seconds() / 60), 1)
        box_height = ppm * minutes

        try:
            artist_name = live["グループ名_採用"]
            if artist_name == "" or artist_name is None:
                artist_name = live["グループ名"]
        except KeyError:
            artist_name = live["グループ名"]
        disp_from = live.get("_display_time_from") or live["ライブステージ"]["from"]
        disp_to = live.get("_display_time_to") or live["ライブステージ"]["to"]
        disp_minutes = _diff_minutes(disp_from, disp_to)
        time_text = f"{disp_from} ～ {disp_to} ({disp_minutes})"
        name_width = font.getlength(artist_name)
        time_width = font.getlength(time_text)

        if one_line_mode:
            line_count = 1
            wrap = False
            # 1 行モードでは時間表記の x 位置を揃えるため、ここでは仮値を入れ、
            # ループ後にグループ名最大幅から共通の required_inner_width を再計算する
            required_inner_width = 0.0
        else:
            # 3 行に伸ばせるか（折り返し可否）
            can_wrap = box_height >= (3 * line_height + 2 * _BOX_VPAD)
            # 元画像縦横比に合わせた場合の想定横幅と比較
            if target_box_width_scaled is not None:
                threshold_w = target_box_width_scaled
            else:
                # フォールバック: 標準縦横比から逆算した想定横幅
                threshold_w = box_height / TARGET_BOX_ASPECT_2LINE
            if can_wrap and name_width > threshold_w:
                line_count = 3
                wrap = True
                required_inner_width = max(name_width / 2.0, time_width) + _WIDTH_SAFETY_PAD
            else:
                line_count = 2
                wrap = False
                required_inner_width = max(name_width, time_width) + _WIDTH_SAFETY_PAD

        per_event.append({
            "live": live,
            "start": start, "end": end, "minutes": minutes,
            "artist_name": artist_name, "time_text": time_text,
            "name_width": name_width, "time_width": time_width,
            "line_count": line_count, "wrap": wrap,
            "required_inner_width": required_inner_width,
        })

    # --- ステップ E: 1 行モードでの時刻列 x オフセット ---
    # 1 行モードでは時間表記の x 位置を全枠で揃える。
    # 列幅 = 全イベントのグループ名最大幅
    oneline_time_x_offset = 0.0
    if one_line_mode:
        max_name_w = max(e["name_width"] for e in per_event)
        max_time_w = max(e["time_width"] for e in per_event)
        oneline_required = max_name_w + spacer_width + max_time_w + _WIDTH_SAFETY_PAD
        oneline_time_x_offset = max_name_w + spacer_width
        for e in per_event:
            e["required_inner_width"] = oneline_required

    # --- ステップ F: 画像横幅決定 ---
    # 重複インデントは出演枠（required_inner_width）に対する比率で動的決定
    base_max_required = max(e["required_inner_width"] for e in per_event)
    duplicate_margin = max(
        _DUPLICATE_MARGIN_MIN,
        int(round(base_max_required * _DUPLICATE_MARGIN_RATIO)),
    )
    # 各 event ごとに「required_inner + 個別 duplicate_margin」を計算し、その最大値を採用
    # （max_required と max_duplicate を別個に加算するより正確）
    max_required_with_dup = 0.0
    duplicate_margin_total = 0
    before_end = per_event[0]["start"]
    for e in per_event:
        if e["start"] < before_end:
            duplicate_margin_total += duplicate_margin
        else:
            duplicate_margin_total = 0
        before_end = max(before_end, e["end"])
        cand = e["required_inner_width"] + duplicate_margin_total
        if cand > max_required_with_dup:
            max_required_with_dup = cand
    # ceil で必ず要求幅以上を確保
    # ここでは MAX_GEN_WIDTH のクランプは行わない。
    # full size でレンダリング後、必要なら画像全体を比率保持で縮小する（ステップ H）。
    image_width = int(math.ceil(
        max_required_with_dup
        + 2 * text_margin
        + 2 * box_margin
        + timeline_text_margin
        + 2 * margin
    ))

    # --- ステップ G: 描画 ---
    text_color = "black"
    background_color = "white"
    line_color = "gray"

    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # 時間軸を描画
    if show_timeline_labels:
        current_y = start_margin
        current_time = start_time
        while current_time <= end_time:
            draw.line(
                [(margin + timeline_text_margin, int(current_y)),
                 (image_width - margin, int(current_y))],
                fill=line_color, width=1,
            )
            draw.text(
                (margin, int(current_y) - line_height // 2),
                current_time.strftime("%H:%M"),
                fill=text_color, font=font,
            )
            current_y += time_line_spacing
            current_time += datetime.timedelta(minutes=30)

    # 各ライブ枠を描画
    duplicate_margin_total = 0
    before_end = per_event[0]["start"]
    for e in per_event:
        if e["start"] < before_end:
            duplicate_margin_total += duplicate_margin
        else:
            duplicate_margin_total = 0
        before_end = max(before_end, e["end"])

        start_y = start_margin + int((e["start"] - start_time).total_seconds() / 60 / 30 * time_line_spacing)
        end_y = start_margin + int((e["end"] - start_time).total_seconds() / 60 / 30 * time_line_spacing)
        box_height = end_y - start_y

        box_left = margin + timeline_text_margin + box_margin + duplicate_margin_total
        box_right = image_width - margin - box_margin
        draw.rectangle(
            [(box_left, start_y), (box_right, end_y)],
            fill=box_color, outline="black",
        )

        max_text_width = max(1, box_right - box_left - 2 * text_margin)
        text_x = box_left + text_margin

        if e["line_count"] == 1:
            # 1 行表示: グループ名と時間を別々に描画し、時間の x 位置を全枠で揃える
            text_height = line_height
            text_start_y = start_y + (box_height - text_height) // 2
            draw.text((text_x, text_start_y), e["artist_name"], fill=text_color, font=font)
            time_x = text_x + int(round(oneline_time_x_offset))
            # 万が一クリップで枠右端を超える場合のフォールバック
            if time_x + e["time_width"] > box_right - text_margin:
                time_x = max(text_x, int(box_right - text_margin - e["time_width"]))
            draw.text((time_x, text_start_y), e["time_text"], fill=text_color, font=font)
        else:
            if e["wrap"]:
                # 3 行: グループ名 (2 行折り返し) + 時間
                name_lines = _wrap_by_pixel(e["artist_name"], font, max_text_width)
                name_lines = _truncate_lines(name_lines, font, max_text_width, MAX_NAME_LINES)
            else:
                # 2 行: グループ名 + 時間（横に収まらない名前は省略）
                name_lines = _wrap_by_pixel(e["artist_name"], font, max_text_width)
                name_lines = _truncate_lines(name_lines, font, max_text_width, 1)
            text_lines = name_lines + [e["time_text"]]
            text_height = line_height * len(text_lines)
            text_start_y = start_y + (box_height - text_height) // 2
            for line in text_lines:
                draw.text((text_x, text_start_y), line, fill=text_color, font=font)
                text_start_y += line_height

    # --- ステップ H: MAX_GEN_WIDTH 超過時は画像全体を比率保持で縮小 ---
    # フォントだけを縮小せず、レンダリング結果を LANCZOS で縮小することで
    # 縦横比・テキスト・枠の見た目の整合性をそのまま維持する。
    if apply_max_width_clamp and image.width > MAX_GEN_WIDTH:
        scale = MAX_GEN_WIDTH / image.width
        new_h = max(1, int(round(image.height * scale)))
        image = image.resize((MAX_GEN_WIDTH, new_h), Image.LANCZOS)

    return image
