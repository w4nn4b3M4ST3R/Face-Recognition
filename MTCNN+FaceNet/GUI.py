import tkinter as tk
import subprocess
import sys
from PIL import Image, ImageTk

proc = None

def check_proc():
    global proc
    if proc and proc.poll() is not None:  # nếu process con thoát
        window.destroy()  # tắt GUI luôn
    else:
        window.after(500, check_proc)

def live_command():
    global proc
    proc = subprocess.Popen([sys.executable, "src/face_rec_cam.py"])
    window.withdraw()

    new_win = tk.Toplevel()
    new_win.geometry("500x250")
    new_win.title("FaceQTH")

    tk.Label(new_win, text="Cửa sổ sẽ hiện ra trong giây lát ...", font=("Arial", 14, "bold")).pack(pady=15)
    tk.Label(new_win, text="Nếu hệ thống nhận diện bạn là Unknown:", font=("Arial", 12)).pack(pady=10)
    tk.Label(new_win, text="- Nhấn n để tạo data mới\n- Nhấn c để chụp 20 ảnh",
             font=("Arial", 12, "italic")).pack(pady=5)
    tk.Label(new_win, text="- Nhấn q để thoát chương trình", font=("Arial", 12), fg="red").pack(pady=0)

    check_proc()

def video_command():
    global proc
    proc = subprocess.Popen([sys.executable, "src/face_rec.py", "--path", "video/video_testing.mp4"])
    window.withdraw()

    new_win = tk.Toplevel()
    new_win.geometry("500x250")
    new_win.title("FaceQTH")

    tk.Label(new_win, text="Cửa sổ sẽ hiện ra trong giây lát ...", font=("Arial", 14, "bold")).pack(pady=15)
    tk.Label(new_win, text="- Nhấn q để thoát chương trình", font=("Arial", 12), fg="red").pack(pady=0)

    check_proc()

def quit_program(event=None):
    global proc
    if proc is not None:
        proc.terminate()
    window.destroy()

# GUI chính
window = tk.Tk()
window.title("FaceQTH")
window.geometry("500x250")
window.configure(bg="white")

# Load ảnh
img = Image.open("logo.png")
img.thumbnail((400, 240), Image.LANCZOS)
photo = ImageTk.PhotoImage(img)

label_img = tk.Label(image=photo, bg="white", borderwidth=0, highlightthickness=0)
label_img.image = photo
label_img.place(x=30, y=10)

button_live = tk.Button(text="Live Mode", height=3, width=15, command=live_command)
button_live.place(x=300, y=50)
button_video = tk.Button(text="Video Mode", height=3, width=15, command=video_command)
button_video.place(x=300, y=150)

# Bind phím tắt
window.bind('q', quit_program)
window.bind('Q', quit_program)

window.mainloop()