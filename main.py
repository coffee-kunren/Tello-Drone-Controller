# <実装機能>
#   基本操作(手動)
#   人間判別機能 + 前方衝突防止センサー
#   最も近い対象への追従機能(その場回転のみ) + 人間追従時、高さを頭の位置でキープ
#   写真・録画 + ピースを合図に3秒後に写真を撮る機能

#   ※写真撮影完了時にドローンを軽く動かして合図を出す要素を追加

import tellopy
import pygame
import cv2
import av
from ultralytics import YOLO
import numpy as np
import datetime
import os
import time
from threading import Thread, Timer
import mediapipe as mp

# ------------------------------
# 標準使用パラメータ
# ------------------------------
FULL_SPEED = 50      # 前後左右の移動スピード(最高速度)
SLOW_SPEED = 15      # 前方警戒時のスピード(制限速度)
YAW_SPEED = 60       # 回転スピード
VERTICAL_SPEED = 60  # 上下移動スピード

# ------------------------------
# 衝突防止 判定用パラメータ
# ------------------------------
SLOW_RATIO = 0.3
STOP_RATIO = 0.5
TARGET = 0           # 判定対象（0：人間、39：ペットボトル）

# ------------------------------
# 中央補正パラメータ
# ------------------------------
CENTER_TOLERANCE_X = 150   # 対象の中心とカメラの中央との差の許容範囲 (横軸)
CENTER_TOLERANCE_Y = 50    #                    〃                    (縦軸)
AUTO_YAW_SPEED = 60        # 自動補正の回転スピード(40以上にすると左右の回転を繰り返しだす)
AUTO_UP_SPEED = 50         # 自動補正の上昇スピード(上昇と下降で速度の影響が異なるため個別に設定)
AUTO_DOWN_SPEED = 1        # 自動補正の下降スピード(値が大きいと上下運動を繰り返しだす)
TOP_MARGIN = 60            # 対象の頭と画面上端の余白(単位：px)(※CENTER_TOLERANCE_Yより大きい値にすること)

# ------------------------------
# カメラの写真・録画
# ------------------------------
class Recorder:
    def __init__(self, output_dir, fps=240):    # 「fps=240」 → Telloの実fpsとOpenCVの再生fps差を補正するため
        self.output_dir = output_dir
        self.fps = fps
        self.video_writer = None
        self.recording = False

    # -- 録画開始 --
    def start(self, frame_size):
        os.makedirs(self.output_dir, exist_ok=True)  # 保存フォルダ(なければ作成)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"tello_{timestamp}.mp4")
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            frame_size
        )
        self.recording = True
        print(f"[通知] 録画開始：{filename}")

    # -- 録画停止 --
    def stop(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("[通知] 録画停止")

    # -- 録画の書き込みフラグ --
    def write(self, frame):
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    # -- 写真を保存 --
    def photo(self, frame):
        os.makedirs(self.output_dir, exist_ok=True)  # 保存フォルダ(なければ作成)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"tello_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[通知] 写真保存：{filename}")

# ------------------------------
# ピースサインクラス(mediapipe)
# ------------------------------
class PeaceCamera:
    def __init__(self, recorder, app, required_time=3, cooldown=3.0, photo_message_callback=None):
        self.recorder = recorder
        self.app = app
        self.required_time = required_time
        self.cooldown = cooldown
        self.state = "IDLE"
        self.peace_start_time = None
        self.countdown_start = None
        self.last_shot_time = 0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,               # カメラ内の手の同時検出数(多くすると処理重くなるかも？)
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 写真保存時に呼ばれるコールバック
        self.photo_message_callback = photo_message_callback

    def is_peace(self, lm):
        index_up  = lm[8].y  < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y > lm[14].y
        pinky_up  = lm[20].y > lm[18].y
        return index_up and middle_up and ring_up and pinky_up

    def notify_drone(self):
        drone = self.app.drone

        # 左右に軽く動いて通知
        try:
            drone.right(60)
            time.sleep(0.5)
            drone.right(0)
            drone.left(60)
            time.sleep(0.5)
            drone.left(0)
        except Exception:
            pass

    def update(self, frame, display_frame):
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        peace_detected = False
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if self.is_peace(hand_landmarks.landmark):
                    peace_detected = True

        # 状態遷移
        if self.state == "IDLE":
            if peace_detected and now - self.last_shot_time > self.cooldown:
                if self.peace_start_time is None:
                    self.peace_start_time = now
                elif now - self.peace_start_time >= self.required_time:
                    self.countdown_start = now
                    self.peace_start_time = None
                    self.state = "COUNTDOWN"
            else:
                self.peace_start_time = None

        if self.state == "COUNTDOWN":
            elapsed = now - self.countdown_start
            remaining = self.required_time - elapsed

            if remaining > 0:
                display_number = int(np.ceil(remaining))  # 切り上げ
                (w, h), _ = cv2.getTextSize(str(display_number), cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
                cv2.putText(display_frame, f"{display_number}",
                            ((frame.shape[1] - w) // 2, (frame.shape[0] + h) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4, (0, 0, 255), 5)
            else:
                # 写真撮影
                self.recorder.photo(frame)
                self.last_shot_time = now
                self.countdown_start = None
                self.state = "IDLE"

                # ドローン側で通知
                self.notify_drone()

                # 写真保存メッセージ用コールバック
                if self.photo_message_callback:
                    self.photo_message_callback()

# ------------------------------
# 映像スレッド(画面出力)
# ------------------------------
class TelloVideo:
    def __init__(self, drone, width, height):
        self.drone = drone
        self.width = width
        self.height = height
        self.frame = None
        self.running = True

    def start(self):
        Thread(target=self._video_thread, daemon=True).start()

    def _video_thread(self):
        try:
            container = av.open(self.drone.get_video_stream(), format='h264', mode='r')
            for packet in container.demux(video=0):
                if not self.running:
                    break
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    img = cv2.resize(img, (self.width, self.height))
                    self.frame = img
        except Exception as e:
            print("[エラー] デコード失敗：", e)
            self.running = False

    def stop(self):
        self.running = False

# ------------------------------
# キー入力(pygame)
# ------------------------------
class InputHandler:
    controls = {
        pygame.K_w: 'forward',                # 前進
        pygame.K_s: 'backward',               # 後退
        pygame.K_a: 'left',                   # 左スライド
        pygame.K_d: 'right',                  # 右スライド
        pygame.K_UP: 'up',                    # 上昇
        pygame.K_DOWN: 'down',                # 下降
        pygame.K_LEFT: 'counter_clockwise',   # 反時計回り
        pygame.K_RIGHT: 'clockwise',          # 時計回り
        pygame.K_TAB: 'takeoff',              # 離陸
        pygame.K_BACKSPACE: 'land',           # 着陸
    }

    special_keys = {
        pygame.K_r: 'record',                 # 録画開始/停止
        pygame.K_RETURN: 'photo',             # 写真撮影
        pygame.K_c: 'centering_toggle',       # 中央補正 ON /S OFF
    }

    def __init__(self, drone, recorder, app):
        self.drone = drone
        self.recorder = recorder
        self.app = app

    def handle_event(self, event, frame):
        if event.type == pygame.KEYDOWN:
            if event.key in self.controls:
                cmd = self.controls[event.key]

                # STOP中の前進入力をブロック
                if self.app.danger_level == "STOP" and cmd in ["forward"]:
                    return    # 他の入力を受け付けないように

                if cmd == 'takeoff':
                    self.drone.takeoff()
                elif cmd == 'land':
                    self.drone.land()
                # 前進入力
                elif cmd == "forward":
                        self.app.is_forwarding = True
                        if self.app.danger_level == "SLOW":
                            self.app.is_slowing = True

                        speed = self.app.get_forward_speed()
                        self.drone.forward(speed)
                # 前進以外の移動入力
                elif cmd in ["backward", "left", "right"]:
                    getattr(self.drone, cmd)(FULL_SPEED)
                # 上昇・下降
                elif cmd in ["up", "down"]:
                    getattr(self.drone, cmd)(YAW_SPEED)
                # 回転入力
                else:
                    getattr(self.drone, cmd)(VERTICAL_SPEED)

            elif event.key in self.special_keys:
                if self.special_keys[event.key] == 'record':
                    if self.recorder.recording:
                        self.recorder.stop()
                    else:
                        self.recorder.start((self.app.video_width, self.app.video_height))

                elif self.special_keys[event.key] == 'photo' and frame is not None:
                    self.recorder.photo(frame)
                    self.app.photo_message_until = time.time() + 2.0    # 2秒間のタイマー

                elif self.special_keys[event.key] == 'centering_toggle':
                    self.app.centering_enabled = not self.app.centering_enabled
                    state = "ON" if self.app.centering_enabled else "OFF"
                    print(f"[通知] 中央補正 {state}")
                    if not self.app.centering_enabled:
                        self.drone.clockwise(0)
                        self.drone.up(0)
                        self.drone.down(0)

        elif event.type == pygame.KEYUP:
            if event.key in self.controls:
                cmd = self.controls[event.key]

                if cmd == "forward":
                    self.app.is_forwarding = False
                    self.app.is_slowing = False
                    self.drone.forward(0)
                if cmd not in ['takeoff', 'land']:
                    getattr(self.drone, cmd)(0)

# ------------------------------
# メイン
# ------------------------------
class TelloApp:
    # -- 写真・録画ファイルの保存フォルダ(同階層に作成、「recordings/20260123_095830」みたいな感じ) --
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "recordings")

    def __init__(self):
        # 画面サイズ
        self.video_width = 640
        self.video_height = 480

        self.drone = tellopy.Tello()

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.BASE_OUTPUT_DIR, now)

        self.recorder = Recorder(output_dir)
        self.video = TelloVideo(self.drone, self.video_width, self.video_height)
        self.battery = None
        self.photo_message_until = 0      # 写真撮影メッセージ用タイマー
        self.yolo = YOLO("yolov8n.pt")    # 人間判別ライブラリ
        self.danger_level = "SAFE"
        self.is_forwarding = False        # 前進中フラグ
        self.is_slowing = False           # 前進中警戒速度フラグ
        self.centering_enabled = False    # 中央補正 ON / OFF
        self.manual_yaw = False           # 左右キー押下中フラグ
        self.manual_vertical = False      # 上下キー押下中フラグ
        self.peace_camera = PeaceCamera(  # ピースサインクラス変数
            recorder=self.recorder,
            app=self,
            required_time=3,
            cooldown=3.0,
            photo_message_callback=lambda: setattr(self, 'photo_message_until', time.time() + 2.0)
        )

    def flight_data_handler(self, event, sender, data):
        self.battery = data.battery_percentage

    # 前進速度を返す
    def get_forward_speed(self):
        if self.danger_level == "STOP":
            return 0
        elif self.danger_level == "SLOW":
            return SLOW_SPEED
        else:
            return FULL_SPEED

    # 前進慣性を打ち消すための短時間の後退命令を出す
    def emergency_stop(self, duration=0.2, speed=FULL_SPEED):
        # 後退命令
        self.drone.forward(0)
        self.drone.backward(speed)


        print("\033[31m-- 後退命令中 -- 後退命令中 --\033[0m")

        # duration 秒後に止める
        def stop():
            self.drone.backward(0)

        Timer(duration, stop).start()

    # メインの代わり
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((320, 330))    # キー操作→pygame 映像→OpenCV
        pygame.display.set_caption("Tello Controls")

        font = pygame.font.SysFont("meiryo", 16)

        lines = [
            "このウィンドウをフォーカスした状態で",
            "ドローンの操作の入力をしてください",
            "",
            "＜操作キー＞",
            "ＷＡＳＤ ：移動(前後左右)",
            " ↑  ↓  ：上昇 / 下降",
            " ←  →  ：回転(左右)",
            "   Tab   ：離陸",
            "Backspace：着陸",
            "   Ｒ    ：録画(開始 / 停止)",
            "  Enter  ：写真撮影",
            "   Ｃ    ：人間追従(ON / OFF)"
        ]

        y = 20
        screen.fill((0, 0, 0))       # 背景クリア
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += font.get_linesize()  # 行間調整

        pygame.display.flip()        # 初回だけ更新

        try:
            self.drone.connect()
            self.drone.wait_for_connection(10.0)
            self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_handler)
            self.drone.start_video()
            print("[通知] Tello接続 カメラ起動")

            self.video.start()
            input_handler = InputHandler(self.drone, self.recorder, self)

            while True:
                frame = self.video.frame
                if frame is not None:
                    # 画面表示用にコピーを作る
                    display_frame = frame.copy()

                    # 画面上の人間を四角で囲って強調表示(YOLO)
                    results = self.yolo(frame)[0]

                    frame_h, frame_w = frame.shape[:2]
                    frame_area = frame_h * frame_w
                    frame_cx = frame_w // 2
                    frame_cy = frame_h // 2

                    targets = []  # (ratio, x1, y1, x2, y2)

                    for box, cls, conf in zip(
                        results.boxes.xyxy,
                        results.boxes.cls,
                        results.boxes.conf
                    ):
                        if int(cls) == TARGET and conf > 0.5:
                            x1, y1, x2, y2 = map(int, box)

                            ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                            targets.append((ratio, x1, y1, x2, y2))

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if targets:
                        ratio, x1, y1, x2, y2 = max(targets, key=lambda t: t[0])
                        target_cx = (x1 + x2) // 2
                        target_cy = (y1 + y2) // 2
                        error_x = target_cx - frame_cx
                        error_y = target_cy - frame_cy
                    else:
                        error_x = error_y = 0

                    # 照準モード時、画面の十字ラインと対象の中心点を表示
                    if self.centering_enabled:
                        cv2.line(display_frame, (frame_cx, 0), (frame_cx, frame_h), (255, 0, 0), 1)
                        cv2.line(display_frame, (0, frame_cy), (frame_w, frame_cy), (255, 0, 0), 1)
                        if targets:
                            cv2.circle(display_frame, (target_cx, target_cy), 5, (0, 0, 255), -1)

                    # 最大占有率から衝突の危険レベルを判定し、画面に表示
                    max_ratio = max(t[0] for t in targets) if targets else 0.0

                    if max_ratio > STOP_RATIO:
                        self.danger_level = "STOP"
                    elif max_ratio > SLOW_RATIO:
                        self.danger_level = "SLOW"
                    else:
                        self.danger_level = "SAFE"

                    color = {
                        "SAFE": (0, 255, 0),
                        "SLOW": (0, 255, 255),
                        "STOP": (0, 0, 255)
                    }[self.danger_level]

                    # 衝突センサー(状態・係数)
                    cv2.putText(display_frame, f"{self.danger_level} ({max_ratio:.2f})",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # バッテリー残量
                    cv2.putText(display_frame, f"Battery {self.battery}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # 録画中のみ「*REC」を表示
                    if self.recorder.recording:
                        cv2.putText(display_frame, "*REC", (frame.shape[1] - 90, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # 写真撮影時2秒間、写真を保存したことを表示
                    if time.time() < self.photo_message_until:
                        cv2.putText(display_frame, "Photo Saved.", (260, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                    # キー押下状態を取得してフラグ更新
                    keys = pygame.key.get_pressed()
                    self.manual_yaw = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
                    self.manual_vertical = keys[pygame.K_UP] or keys[pygame.K_DOWN]

                    # カメラ中央補正（追尾アシスト）
                    if self.centering_enabled and targets:
                        if not self.manual_yaw:
                            # 左右回転補正
                            if abs(error_x) > CENTER_TOLERANCE_X:
                                if error_x > 0:
                                    self.drone.clockwise(AUTO_YAW_SPEED)
                                else:
                                    self.drone.counter_clockwise(AUTO_YAW_SPEED)
                            else:
                                self.drone.counter_clockwise(0)
                                self.drone.clockwise(0)

                        if not self.manual_vertical:
                            # 追尾対象の最上部を取得（y1が上端）
                            _, x1, y1, _, _ = max(targets, key=lambda t: t[0])  # 最大面積の対象

                            # 画面中央補正ではなく上端基準
                            desired_y = TOP_MARGIN
                            error_y_top = y1 - desired_y

                            # 上下補正
                            if abs(error_y_top) > CENTER_TOLERANCE_Y:
                                if error_y_top > 0:
                                    # self.drone.down(AUTO_DOWN_SPEED)
                                    self.drone.up(AUTO_DOWN_SPEED)
                                else:
                                    self.drone.up(AUTO_UP_SPEED)
                            else:
                                self.drone.up(0)
                                self.drone.down(0)

                    # カメラ衝突判定と前進速度関係
                    if self.is_forwarding:
                        if self.danger_level == "STOP":
                            print("\033[33m-- 止まります --\033[0m")
                            self.emergency_stop(duration=0.2, speed=FULL_SPEED)
                            self.is_forwarding = False  # 停止後は前進フラグ解除
                            self.is_slowing = False     # 同様に前進警戒フラグ解除
                        elif self.danger_level == "SLOW" and not self.is_slowing:
                            self.drone.forward(SLOW_SPEED)
                            self.is_slowing = True
                        elif self.danger_level == "SAFE" and self.is_slowing:
                            self.drone.forward(FULL_SPEED)
                            self.is_slowing = False

                    # ピースサイン判定・撮影
                    self.peace_camera.update(frame, display_frame)

                    # 録画フラグ確認 + 画面表示(※最後に行うこと！)
                    self.recorder.write(frame)
                    cv2.imshow("Tello Camera", display_frame)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    input_handler.handle_event(e, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # time.sleep(0.01)    # ←CPU負荷軽減対策

        except KeyboardInterrupt:
            print("[通知] プログラムを終了します...")
        finally:
            self.drone.land()
            self.recorder.stop()
            self.video.stop()
            self.drone.quit()
            cv2.destroyAllWindows()
            pygame.quit()

# ------------------------------
# 実行
# ------------------------------
if __name__ == "__main__":
    TelloApp().run()
