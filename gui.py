from tkinter import *

top = Tk()
top.geometry("300x300")
top["bg"] = "green"
top.title("Fire Recognition")

def red():
    new_window = Toplevel()
    new_window.geometry("300x300")
    new_window["bg"] = "#A4DE02"
    
redbutton = Button(top, text="Red", fg="red", justify="center", command=red)
redbutton.pack()

current_label1 = Label(top, text="FireRecognition System",
                        justify='center', width=45, background="Yellow", foreground="blue")
current_label1.pack()

button = Button(top, text="Click", command=red)
button.pack()

top.mainloop()
