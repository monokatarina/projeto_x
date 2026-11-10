import os
import time
import math
import argparse
import random
import numpy as np
import cv2
import serial
from serial.tools import list_ports
from ultralytics import YOLO
import mss

# ════════════════ CONFIGURAÇÕES DE PROCESSAMENTO ════════════════
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "last.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(MODEL_DIR, "last.pt")

FOV_RADIUS = 150 
CONFIDENCE = 0.60
IMGSZ      = 256

# ── Ajustes de Entrada e Movimento ──
PIXEL_TO_HARDWARE = 8  
AIM_SPEED = 0.85      
DEADZONE  = 2              
PACKET_DELAY = 0.002       
MAX_JUMP_PX = 65.0         

# ── Filtros de Suavização ──
HUMAN_SWAY = 0.05         
BEZIER_SMOOTHING = 0.8      
SPEED_NOISE = 0.0       

# ── Seleção de ROI e Tolerâncias ──
HEAD_PRIORITY_MULT = 9999.0  
MICRO_FLICK_ERROR = 0.0  
DYNAMIC_SCALE_ENABLED = False

# ── Persistência de Seleção ──
STICKY_RADIUS = 160.0    
LOCK_TIMEOUT = 0.400  

# ════════════════ CONTROLADOR DE FLUXO VETORIAL ════════════════
class MotionController:
    def __init__(self):
        self.rem_x = 0.0
        self.rem_y = 0.0
        self.current_sway_x = 0.0
        self.current_sway_y = 0.0
        self.target_sway_x = 0.0
        self.target_sway_y = 0.0
        self.last_sway_time = time.time()
        
        self.overshoot_x = 0.0
        self.overshoot_y = 0.0

    def compute_step(self, dx, dy):
        dist = math.hypot(dx, dy)
        if dist <= DEADZONE:
            self.rem_x, self.rem_y = 0.0, 0.0
            return 0, 0
            
        now = time.time()
        
        if now - self.last_sway_time > 0.15: 
            self.target_sway_x = random.uniform(-HUMAN_SWAY, HUMAN_SWAY)
            self.target_sway_y = random.uniform(-HUMAN_SWAY, HUMAN_SWAY)
            self.last_sway_time = now
            
        self.current_sway_x += (self.target_sway_x - self.current_sway_x) * BEZIER_SMOOTHING
        self.current_sway_y += (self.target_sway_y - self.current_sway_y) * BEZIER_SMOOTHING

        if MICRO_FLICK_ERROR > 0:
            if dist > 35 and abs(self.overshoot_x) < 1:
                self.overshoot_x = random.choice([-1, 1]) * random.uniform(MICRO_FLICK_ERROR * 0.5, MICRO_FLICK_ERROR)
                self.overshoot_y = random.choice([-1, 1]) * random.uniform(MICRO_FLICK_ERROR * 0.5, MICRO_FLICK_ERROR)
            
            if dist < 25:
                self.overshoot_x *= 0.70  
                self.overshoot_y *= 0.70
        else:
            self.overshoot_x = 0.0
            self.overshoot_y = 0.0

        target_dx = dx + self.overshoot_x
        target_dy = dy + self.overshoot_y
        
        if dist > 20:
            target_dx += (dy * self.current_sway_x)
            target_dy += (-dx * self.current_sway_y)
            
        dist_curved = math.hypot(target_dx, target_dy)
        if dist_curved > MAX_JUMP_PX:
            scale = MAX_JUMP_PX / dist_curved
            target_dx *= scale
            target_dy *= scale
            dist_curved = MAX_JUMP_PX
        
        noise = random.uniform(1.0 - SPEED_NOISE, 1.0 + SPEED_NOISE)
        
        if dist < 15:
            current_speed = AIM_SPEED * 0.3 * noise
        elif dist < 45:
            current_speed = AIM_SPEED * 0.6 * noise
        elif dist < 80:
            current_speed = AIM_SPEED * 0.8 * noise
        else:
            current_speed = AIM_SPEED * noise

        dynamic_scale = PIXEL_TO_HARDWARE
        if DYNAMIC_SCALE_ENABLED:
            if dist > 200:
                dynamic_scale *= 1.07
            elif dist > 120:
                dynamic_scale *= 1.04
            elif dist > 80:
                dynamic_scale *= 1.02
            elif dist < 15:
                dynamic_scale *= 0.95

        float_x = (target_dx * current_speed * dynamic_scale) + self.rem_x
        float_y = (target_dy * current_speed * dynamic_scale) + self.rem_y

        step_x = int(round(float_x))
        step_y = int(round(float_y))

        self.rem_x = float_x - step_x
        self.rem_y = float_y - step_y

        return step_x, step_y

# ════════════════ GERENCIADOR DE RASTREAMENTO E PERSISTÊNCIA ════════════════
class TrackerManager:
    def __init__(self, match_radius=STICKY_RADIUS):
        self.match_radius = match_radius
        self.last_pos = None  
        self.last_time = 0.0

    def select_target(self, candidates, now):
        if not candidates:
            if self.last_pos is not None and (now - self.last_time) > LOCK_TIMEOUT:
                self.last_pos = None
            return None
            
        if self.last_pos is not None and (now - self.last_time) <= LOCK_TIMEOUT:
            best_candidate = None
            best_dist = self.match_radius 
            
            for c in candidates:
                d = math.hypot(c["hx"] - self.last_pos[0], c["hy"] - self.last_pos[1])
                if d < best_dist:
                    best_dist = d
                    best_candidate = c
                    
            if best_candidate is not None:
                self.last_pos = (best_candidate["hx"], best_candidate["hy"])
                self.last_time = now
                return best_candidate
                
        best_candidate = min(candidates, key=lambda c: math.hypot(c["dx"], c["dy"]))
        self.last_pos = (best_candidate["hx"], best_candidate["hy"])
        self.last_time = now
        return best_candidate

# ════════════════ COMUNICAÇÃO SERIAL DE HARDWARE ════════════════
def auto_detect_port():
    for p in list_ports.comports():
        if any(k in f"{p.device} {p.description} {p.hwid}".lower() for k in ("leonardo", "arduino", "usb serial", "ch340")):
            return p.device
    return None

def send_serial_command(ser, dx_steps, dy_steps):
    if not ser or (dx_steps == 0 and dy_steps == 0):
        return
    rem_x, rem_y = dx_steps, dy_steps
    while rem_x != 0 or rem_y != 0:
        step_x = max(-127, min(127, int(rem_x)))
        step_y = max(-127, min(127, int(rem_y)))
        try:
            ser.write(f"M {step_x} {step_y}\n".encode("ascii"))
            time.sleep(PACKET_DELAY) 
        except Exception:
            pass
        rem_x -= step_x
        rem_y -= step_y

# ════════════════ ANÁLISE DE GEOMETRIA ════════════════
def extract_regions(frame, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        cx, cy = x1 + w // 2, y1 + h // 2
        return cx, cy, cx, cy, (x1, y1, x2, y2)
        
    base_cx = x1 + (w // 2)
    
    if h < 110:
        offset = 0.13
    elif h < 175:
        offset = 0.12
    else:
        offset = 0.12

    top_cx = base_cx
    top_cy = y1 + int(h * offset)
    
    mid_x = base_cx
    mid_y = y1 + int(h * 0.35) 
    
    radius = max(5, int(h * 0.08))
    box_region = (top_cx - radius, top_cy - radius, 
                  top_cx + radius, top_cy + radius)
    
    return top_cx, top_cy, mid_x, mid_y, box_region

# ════════════════ EXECUÇÃO PRINCIPAL ════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Exibe janela de visualização")
    args = parser.parse_args()

    ser = None
    port = auto_detect_port()
    if port:
        try:
            ser = serial.Serial(port, 115200, timeout=0, write_timeout=0)
            time.sleep(2)
            ser.reset_input_buffer()
        except Exception:
            pass

    model = YOLO(MODEL_PATH)
    names = model.names
    target_class_ids = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if str(n).lower() == "enemy"] or None
    
    sct = mss.MSS()
    monitor = sct.monitors[1]
    cx, cy = monitor["width"] // 2, monitor["height"] // 2
    capture_size = FOV_RADIUS * 2
    half_cap = capture_size / 2.0
    region = {
        "left": cx - FOV_RADIUS, "top": cy - FOV_RADIUS, 
        "width": capture_size, "height": capture_size
    }

    motion_controller = MotionController()
    tracker = TrackerManager() 
    t0 = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            try:
                shot = sct.grab(region)
                frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((capture_size, capture_size, 4))[:, :, :3].copy()
                results = model(frame, verbose=False, conf=CONFIDENCE, imgsz=IMGSZ, classes=target_class_ids, device="cpu")
                candidates = []

                if len(results) > 0 and results[0].boxes is not None:
                    xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    for x1, y1, x2, y2 in xyxy:
                        if x2 <= x1 or y2 <= y1: continue
                        
                        top_x, top_y, mid_x, mid_y, box_reg = extract_regions(frame, x1, y1, x2, y2)
                        dist_top = math.hypot(top_x - half_cap, top_y - half_cap)
                        dist_mid = math.hypot(mid_x - half_cap, mid_y - half_cap)
                        
                        if dist_top <= (dist_mid * HEAD_PRIORITY_MULT):
                            target_x, target_y = top_x, top_y
                            zone = "ZONE_A"
                        else:
                            target_x, target_y = mid_x, mid_y
                            zone = "ZONE_B"

                        dx = target_x - half_cap
                        dy = target_y - half_cap
                        
                        candidates.append({
                            "dx": dx, "dy": dy, "hx": target_x, "hy": target_y, 
                            "box": (x1, y1, x2, y2), "hb": box_reg, "zone": zone,
                            "top": (top_x, top_y), "mid": (mid_x, mid_y)
                        })

                now = time.time()
                best_target = tracker.select_target(candidates, now)

                if best_target is not None:
                    step_x, step_y = motion_controller.compute_step(best_target["dx"], best_target["dy"])
                    if step_x != 0 or step_y != 0:
                        send_serial_command(ser, step_x, step_y)
                else:
                    motion_controller.rem_x, motion_controller.rem_y = 0.0, 0.0

                frames += 1
                if time.time() - t0 >= 1.0:
                    fps = frames / (time.time() - t0)
                    t0 = time.time()
                    frames = 0

                if args.preview:
                    for c in candidates:
                        is_locked = (best_target and c == best_target)
                        color_box = (0, 255, 255) if is_locked else (0, 0, 255)
                        cv2.rectangle(frame, (c["box"][0], c["box"][1]), (c["box"][2], c["box"][3]), color_box, 1)
                        
                        if "ZONE_A" in c["zone"]:
                            cv2.rectangle(frame, c["hb"][:2], c["hb"][2:], (0, 255, 0), 1)
                            cv2.circle(frame, c["top"], 3, (0, 255, 0), -1)
                            cv2.circle(frame, c["mid"], 2, (100, 100, 100), 1)
                        else:
                            cv2.circle(frame, c["top"], 2, (100, 100, 100), 1)
                            cv2.circle(frame, c["mid"], 3, (0, 165, 255), -1)

                    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            except Exception:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        if ser:
            try: ser.write(b"S\n")
            except Exception: pass
            ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()