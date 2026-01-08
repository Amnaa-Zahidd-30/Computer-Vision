import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root=root
        self.root.title("CV Image Processing Tool")
        self.root.geometry("1200x850")
        self.root.config(bg="#2b2b2b")
        
        self.originalimg=None
        self.processedimg=None
        
        self.setupui()
    
    def setupui(self):
        topframe=tk.Frame(self.root, bg="#2b2b2b")
        topframe.pack(pady=10)
        
        tk.Button(topframe, text="Load Image", command=self.loadimg, 
                 bg="#4CAF50", fg="white", width=12, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(topframe, text="Save Image", command=self.saveimg,
                 bg="#2196F3", fg="white", width=12, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(topframe, text="Show Stats", command=self.showstats,
                 bg="#FF9800", fg="white", width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        imgframe=tk.Frame(self.root, bg="#2b2b2b")
        imgframe.pack(pady=10)
        
        left=tk.Frame(imgframe, bg="#404040")
        left.pack(side=tk.LEFT, padx=10)
        tk.Label(left, text="Original", bg="#404040", fg="white", font=("Arial", 12)).pack()
        self.canvas1=tk.Canvas(left, width=450, height=350, bg="black")
        self.canvas1.pack()
        
        right=tk.Frame(imgframe, bg="#404040")
        right.pack(side=tk.LEFT, padx=10)
        tk.Label(right, text="Processed", bg="#404040", fg="white", font=("Arial", 12)).pack()
        self.canvas2=tk.Canvas(right, width=450, height=350, bg="black")
        self.canvas2.pack()
        
        btnframe=tk.Frame(self.root, bg="#2b2b2b")
        btnframe.pack(pady=20)
        
        operations=[
            ("Grayscale", self.grayscale),
            ("Blur", self.blur),
            ("Threshold", self.threshold),
            ("Rotate 45°", lambda: self.rotate(45)),
            ("Rotate 90°", lambda: self.rotate(90)),
            ("Resize 50%", self.resize),
            ("Edge Detection", self.edgedetect),
            ("Sharpen", self.sharpen),
            ("Crop Image", self.crop),
            ("Add Text", self.addtext),
            ("Blend Images", self.blend),
            ("Histogram Eq.", self.histogram)
        ]
        
        for i, (text, cmd) in enumerate(operations):
            row=i // 4
            col=i % 4
            tk.Button(btnframe, text=text, command=cmd, width=15, height=2,
                     bg="#555555", fg="white").grid(row=row, column=col, padx=5, pady=5)
    
    def loadimg(self):
        path=filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.originalimg=cv2.imread(path)
            self.processedimg=self.originalimg.copy()
            self.showimgs()
    
    def showimgs(self):
        if self.originalimg is not None:
            self.displaycanvas(self.canvas1, self.originalimg)
        if self.processedimg is not None:
            self.displaycanvas(self.canvas2, self.processedimg)
    
    def displaycanvas(self, canvas, img):
        if len(img.shape)==2:
            imgrgb=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w=imgrgb.shape[:2]
        maxw, maxh=450, 350
        
        if w>maxw or h>maxh:
            scale=min(maxw/w, maxh/h)
            neww, newh=int(w*scale), int(h*scale)
            imgrgb=cv2.resize(imgrgb, (neww, newh))
        
        imgpil=Image.fromarray(imgrgb)
        imgtk=ImageTk.PhotoImage(imgpil)
        canvas.delete("all")
        canvas.create_image(225, 175, image=imgtk)
        canvas.image=imgtk
    
    def grayscale(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        self.processedimg=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2GRAY)
        self.showimgs()
    
    def blur(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        self.processedimg=cv2.GaussianBlur(self.originalimg, (15, 15), 0)
        self.showimgs()
    
    def threshold(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        gray=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2GRAY)
        _, self.processedimg=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.showimgs()
    
    def rotate(self, angle):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        h, w=self.originalimg.shape[:2]
        center=(w//2, h//2)
        matrix=cv2.getRotationMatrix2D(center, angle, 1.0)
        self.processedimg=cv2.warpAffine(self.originalimg, matrix, (w, h))
        self.showimgs()
    
    def resize(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        
        h, w=self.originalimg.shape[:2]
        neww=w//2
        newh=h//2
        self.processedimg=cv2.resize(self.originalimg, (neww, newh))
        self.showimgs()
        
        messagebox.showinfo("Resize Complete", 
                          f"Original: {w}x{h} pixels\nResized: {neww}x{newh} pixels\n\n50% reduction applied!")
    
    def edgedetect(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        gray=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2GRAY)
        self.processedimg=cv2.Canny(gray, 100, 200)
        self.showimgs()
    
    def sharpen(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        kernel=np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        self.processedimg=cv2.filter2D(self.originalimg, -1, kernel)
        self.showimgs()
    
    def crop(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        
        self.cropwindow=tk.Toplevel(self.root)
        self.cropwindow.title("Crop Image - Click and Drag")
        self.cropwindow.config(bg="#2b2b2b")
        
        tk.Label(self.cropwindow, text="Click and drag to select crop area", 
                bg="#2b2b2b", fg="white", font=("Arial", 12)).pack(pady=10)
        
        self.cropcanvas=tk.Canvas(self.cropwindow, bg="black")
        self.cropcanvas.pack(padx=10, pady=10)
        
        imgrgb=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2RGB)
        h, w=imgrgb.shape[:2]
        maxw, maxh=800, 600
        
        if w>maxw or h>maxh:
            self.cropscale=min(maxw/w, maxh/h)
            neww, newh=int(w*self.cropscale), int(h*self.cropscale)
            imgrgb=cv2.resize(imgrgb, (neww, newh))
        else:
            self.cropscale=1.0
        
        self.cropcanvas.config(width=imgrgb.shape[1], height=imgrgb.shape[0])
        
        imgpil=Image.fromarray(imgrgb)
        self.croptk=ImageTk.PhotoImage(imgpil)
        self.cropcanvas.create_image(0, 0, anchor=tk.NW, image=self.croptk)
        
        self.startx=None
        self.starty=None
        self.rect=None
        
        def onpress(event):
            self.startx=event.x
            self.starty=event.y
            if self.rect:
                self.cropcanvas.delete(self.rect)
            self.rect=self.cropcanvas.create_rectangle(self.startx, self.starty, 
                                                       self.startx, self.starty, 
                                                       outline="red", width=2)
        
        def ondrag(event):
            if self.rect:
                self.cropcanvas.coords(self.rect, self.startx, self.starty, event.x, event.y)
        
        def onrelease(event):
            endx=event.x
            endy=event.y
            
            x1=int(min(self.startx, endx)/self.cropscale)
            y1=int(min(self.starty, endy)/self.cropscale)
            x2=int(max(self.startx, endx)/self.cropscale)
            y2=int(max(self.starty, endy)/self.cropscale)
            
            x1=max(0, x1)
            y1=max(0, y1)
            x2=min(self.originalimg.shape[1], x2)
            y2=min(self.originalimg.shape[0], y2)
            
            if x2>x1 and y2>y1:
                self.processedimg=self.originalimg[y1:y2, x1:x2]
                self.showimgs()
                self.cropwindow.destroy()
            else:
                messagebox.showwarning("Warning", "Please select a valid area!")
        
        self.cropcanvas.bind("<ButtonPress-1>", onpress)
        self.cropcanvas.bind("<B1-Motion>", ondrag)
        self.cropcanvas.bind("<ButtonRelease-1>", onrelease)
    
    def addtext(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        
        textwindow=tk.Toplevel(self.root)
        textwindow.title("Add Text to Image")
        textwindow.geometry("750x550")
        textwindow.config(bg="#1a1a1a")
        
        header=tk.Frame(textwindow, bg="#2d2d2d", height=45)
        header.pack(fill=tk.X)
        tk.Label(header, text="Add Text to Image", bg="#2d2d2d", 
                fg="white", font=("Arial", 13, "bold")).pack(pady=10)
        
        container=tk.Frame(textwindow, bg="#1a1a1a")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        leftside=tk.Frame(container, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        leftside.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        tk.Label(leftside, text="Click on image to place text", bg="#2d2d2d",
                fg="#aaaaaa", font=("Arial", 9)).pack(pady=6)
        
        previewcanvas=tk.Canvas(leftside, bg="black", width=380, height=380)
        previewcanvas.pack(padx=8, pady=8)
        
        imgrgb=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2RGB)
        h, w=imgrgb.shape[:2]
        maxw, maxh=380, 380
        
        if w>maxw or h>maxh:
            self.textscale=min(maxw/w, maxh/h)
            neww, newh=int(w*self.textscale), int(h*self.textscale)
            imgrgb=cv2.resize(imgrgb, (neww, newh))
        else:
            self.textscale=1.0
        
        imgpil=Image.fromarray(imgrgb)
        self.texttk=ImageTk.PhotoImage(imgpil)
        previewcanvas.create_image(0, 0, anchor=tk.NW, image=self.texttk)
        
        clickedpos=[None, None]
        
        def onclick(event):
            clickedpos[0]=int(event.x/self.textscale)
            clickedpos[1]=int(event.y/self.textscale)
            xentry.delete(0, tk.END)
            xentry.insert(0, str(clickedpos[0]))
            yentry.delete(0, tk.END)
            yentry.insert(0, str(clickedpos[1]))
        
        previewcanvas.bind("<Button-1>", onclick)
        
        rightside=tk.Frame(container, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        rightside.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        
        canvas=tk.Canvas(rightside, bg="#2d2d2d", width=280, highlightthickness=0)
        scrollbar=tk.Scrollbar(rightside, orient="vertical", command=canvas.yview)
        scrollframe=tk.Frame(canvas, bg="#2d2d2d")
        
        scrollframe.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollframe, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        tk.Label(scrollframe, text="Text Content", bg="#2d2d2d", fg="white",
                font=("Arial", 10, "bold")).pack(pady=(8, 3))
        
        textentry=tk.Entry(scrollframe, width=30, font=("Arial", 9), bg="#404040",
                          fg="white", insertbackground="white", relief=tk.FLAT, bd=5)
        textentry.pack(pady=3, ipady=2)
        textentry.insert(0, "Your Text Here")
        
        tk.Label(scrollframe, text="Position", bg="#2d2d2d", fg="white",
                font=("Arial", 10, "bold")).pack(pady=(8, 3))
        
        posframe=tk.Frame(scrollframe, bg="#2d2d2d")
        posframe.pack(pady=3)
        
        tk.Label(posframe, text="X:", bg="#2d2d2d", fg="white", font=("Arial", 9)).grid(row=0, column=0, padx=3)
        xentry=tk.Entry(posframe, width=7, font=("Arial", 9), bg="#404040", fg="white")
        xentry.grid(row=0, column=1, padx=3)
        xentry.insert(0, "50")
        
        tk.Label(posframe, text="Y:", bg="#2d2d2d", fg="white", font=("Arial", 9)).grid(row=0, column=2, padx=3)
        yentry=tk.Entry(posframe, width=7, font=("Arial", 9), bg="#404040", fg="white")
        yentry.grid(row=0, column=3, padx=3)
        yentry.insert(0, "100")
        
        tk.Label(scrollframe, text="Font Style", bg="#2d2d2d", fg="white",
                font=("Arial", 10, "bold")).pack(pady=(8, 3))
        
        fontvar=tk.StringVar(value="Regular")
        fonts=[
            ("Regular", cv2.FONT_HERSHEY_SIMPLEX),
            ("Bold", cv2.FONT_HERSHEY_DUPLEX),
            ("Italic", cv2.FONT_HERSHEY_COMPLEX),
            ("Script", cv2.FONT_HERSHEY_SCRIPT_SIMPLEX),
            ("Fancy", cv2.FONT_HERSHEY_SCRIPT_COMPLEX)
        ]
        
        fontframe=tk.Frame(scrollframe, bg="#2d2d2d")
        fontframe.pack(pady=3)
        
        for i, (name, val) in enumerate(fonts):
            row=i//2
            col=i%2
            tk.Radiobutton(fontframe, text=name, variable=fontvar, value=name,
                          bg="#2d2d2d", fg="white", selectcolor="#555555",
                          activebackground="#2d2d2d", activeforeground="white",
                          font=("Arial", 9)).grid(row=row, column=col, sticky="w", padx=8, pady=1)
        
        tk.Label(scrollframe, text="Font Size", bg="#2d2d2d", fg="white",
                font=("Arial", 10, "bold")).pack(pady=(8, 3))
        
        sizevar=tk.IntVar(value=2)
        sizescale=tk.Scale(scrollframe, from_=1, to=5, orient=tk.HORIZONTAL,
                          variable=sizevar, bg="#404040", fg="white", length=200,
                          troughcolor="#555555", highlightthickness=0)
        sizescale.pack(pady=3)
        
        tk.Label(scrollframe, text="Text Color", bg="#2d2d2d", fg="white",
                font=("Arial", 10, "bold")).pack(pady=(8, 3))
        
        colorvar=tk.StringVar(value="White")
        colors=[
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("White", (255, 255, 255)),
            ("Black", (0, 0, 0))
        ]
        
        colorframe=tk.Frame(scrollframe, bg="#2d2d2d")
        colorframe.pack(pady=3)
        
        for i, (name, rgb) in enumerate(colors):
            row=i//2
            col=i%2
            tk.Radiobutton(colorframe, text=name, variable=colorvar, value=name,
                          bg="#2d2d2d", fg="white", selectcolor="#555555",
                          activebackground="#2d2d2d", activeforeground="white",
                          font=("Arial", 9)).grid(row=row, column=col, sticky="w", padx=8, pady=1)
        
        def applytext():
            try:
                text=textentry.get()
                x=int(xentry.get())
                y=int(yentry.get())
                
                fontmap={name: val for name, val in fonts}
                colormap={name: rgb for name, rgb in colors}
                
                font=fontmap[fontvar.get()]
                size=sizevar.get()
                color=colormap[colorvar.get()]
                
                self.processedimg=self.originalimg.copy()
                cv2.putText(self.processedimg, text, (x, y), font, size, color, 2)
                self.showimgs()
                textwindow.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", "Please enter valid values!")

        tk.Button(scrollframe, text="Add Text", command=applytext,
                 bg="#4CAF50", fg="white", width=20, height=2,
                 font=("Arial", 10, "bold"), relief=tk.FLAT).pack(pady=12)
    
    def drawshapes(self):
        pass
    
    def blend(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        path=filedialog.askopenfilename(title="Select second image",
                                         filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            img2=cv2.imread(path)
            h, w=self.originalimg.shape[:2]
            img2=cv2.resize(img2, (w, h))
            self.processedimg=cv2.add(self.originalimg, img2)
            self.showimgs()
    
    def histogram(self):
        if self.originalimg is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        gray=cv2.cvtColor(self.originalimg, cv2.COLOR_BGR2GRAY)
        self.processedimg=cv2.equalizeHist(gray)
        self.showimgs()
    
    def showstats(self):
        if self.processedimg is None:
            messagebox.showwarning("Warning", "Process an image first!")
            return
        
        img=self.processedimg
        if len(img.shape)==3:
            df=pd.DataFrame(img.reshape(-1, 3), columns=['B', 'G', 'R'])
        else:
            df=pd.DataFrame(img.reshape(-1, 1), columns=['Gray'])
        
        statswindow=tk.Toplevel(self.root)
        statswindow.title("Pixel Statistics")
        statswindow.geometry("500x400")
        statswindow.config(bg="#2b2b2b")
        
        tk.Label(statswindow, text="Pixel Statistics", font=("Arial", 14),
                bg="#2b2b2b", fg="white").pack(pady=10)
        
        text=tk.Text(statswindow, bg="#404040", fg="white", font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, df.describe().to_string())
        text.config(state=tk.DISABLED)
    
    def saveimg(self):
        if self.processedimg is None:
            messagebox.showwarning("Warning", "No processed image to save!")
            return
        path=filedialog.asksaveasfilename(defaultextension=".png",
                                           filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.processedimg)
            messagebox.showinfo("Success", "Image saved successfully!")

if __name__=="__main__":
    root=tk.Tk()
    app=ImageProcessor(root)
    root.mainloop()
