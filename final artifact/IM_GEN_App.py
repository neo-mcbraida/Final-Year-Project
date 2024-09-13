from tkinter import *
import customtkinter
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from functools import partial
import os
from tkinter import filedialog

l_frame_width = 600
frame_height = 600
r_frame_width = 400
time_steps = 20
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

absolute_path = os.path.dirname(__file__)

class Image_Frame():
    def __init__(self, model, parent_frame, previous_ims, old_im_frames, images) -> None:
        """
        Initialize ImageFrame instance.

        Args:
            model: TensorFlow model for image generation.
            parent_frame: Parent frame for placing image and buttons.
            previous_ims: List to store previous images.
            old_im_frames: List to store frames for previous images.
            images: List to store current images.
        """
        self.model = model
        self.parent_frame = parent_frame
        self.previous_ims = previous_ims # these are lists so the are mutable, so this works
        self.old_im_frames = old_im_frames
        self.images = images

        # Create image frame
        self.image_frame = customtkinter.CTkFrame(parent_frame,
                                        width=r_frame_width,
                                        height=r_frame_width,
                                        corner_radius=0)
        self.image_frame.grid(row=0,  column=0)
        self.image_frame.grid_propagate(0)
        
        # Create input frame
        self.input_frame = customtkinter.CTkFrame(parent_frame,
                            width=l_frame_width,
                            height=frame_height,
                            corner_radius=10)
        self.input_frame.grid(row=1,  column=0,  padx=10,  pady=5)
        
        # Initialize image label, which will display the generated image
        self.img = ""
        self.image_label = customtkinter.CTkLabel(self.image_frame, text="")
        self.image_label.grid(row=0, column=0)

        # Initialize buttons
        self.printButton = customtkinter.CTkButton(master=self.input_frame,
                                        text="Generate Image")
        self.printButton.grid(row=0, column=0, padx=5, pady=5)
        self.save = customtkinter.CTkButton(master=self.input_frame,
                                        text="Save", 
                                        command=partial(self.save_image))
        self.save.grid(row=0, column=1, padx=5, pady=5) 
        self.progressbar = None
        
    
    def set_image_element(self):        
        """
        Set the image element to the label.
        """
        self.img = self.img.resize((r_frame_width,r_frame_width))
        if self.progressbar != None:
            self.progressbar.destroy()
        self.image_id = self.image_label.configure(image=ImageTk.PhotoImage(self.img))

    def update_image_bar(self):        
        """
        Update the image bar with new image.
        """
        self.images.insert(0, self.img)
        # ensure we have enough image frames to fit the number of images before adding a new image to the list, if not it adds new buttons to fit the image,
        # then it shifts the images along by one
        if len(self.images) > len(self.old_im_frames):
            c = len(self.old_im_frames)
            for i in range(2):
                img_lbl= customtkinter.CTkButton(self.left_frame, 
                                        width=80,
                                        height=80,
                                        text="",
                                        corner_radius=0,
                                        fg_color="#333332",
                                        image="",
                                        command=partial(self.set_main_img, c+i))
                img_lbl.grid(row=((c+i)//2), column=(c+i)%2, padx=10, pady=10)
                self.old_im_frames.append(img_lbl)

        for i in range(len(self.images)):
            # Update the previously generated images on the left side of the screen
            self.old_im_frames[i].configure(image=ImageTk.PhotoImage(self.images[i].resize((80,80))), command=partial(self.set_main_img, i))  

    def set_main_img(self, ind):
        if ind < len(self.images):
            self.img = self.images[ind]
            self.set_image_element()

    def cvtImg(self, img):        
        """
        Convert image.

        Args:
            img: Image to convert.
        """
        img = img - img.min()
        img = (img / img.max())
        return (img * 255).astype(np.uint8)
        
    def save_image(self):        
        """
        Save the current image.
        """
        if self.img != None:
            filename = filedialog.asksaveasfilename(
                    filetypes=(
                        ("JPG Image", "*.jpg"),
                        ("All Files", "*.*")
                    )
            )
            self.img.save(filename+".jpg")
            
class Gan_Image_Frame(Image_Frame):
    def __init__(self, model, parent_frame, previous_ims, old_im_frames, images):
        super().__init__(model, parent_frame, previous_ims, old_im_frames, images)
        self.printButton.configure(command=partial(self.show_image))
        
    def show_image(self):        
        """
        Show generated image.
        In the case of a GAN no progress bar is needed because the generation is so fast, therefor we just call load()
        """
        self.load()
        
class Colour_GAN(Gan_Image_Frame):
    def __init__(self, model, parent_frame, previous_ims, old_im_frames, images):
        super().__init__(model, parent_frame, previous_ims, old_im_frames, images)
        
    def load(self, i=1):
        """
        Generates an image using the colour gan, and updates the necessary elements of the UI
        """
        noise = np.random.normal(size=(1, 100))
        self.img = self.model.predict(noise, verbose=0)
        self.img = Image.fromarray(self.cvtImg(self.img[0]))

        # add image to set of all images (on left of screen)
        self.update_image_bar()

        # set main image to be the one just generated
        self.set_image_element()
        
class BnW_GAN(Gan_Image_Frame):
    def __init__(self, model, parent_frame, previous_ims, old_im_frames, images):
        super().__init__(model, parent_frame, previous_ims, old_im_frames, images)
    
    def load(self, i=1):       
        """
        Load and generate black and white image.
        """
        noise = np.random.normal(size=(1, 100))
        self.img = self.model.predict(noise, verbose=0)
        
        # must be reshaped since to meet the requirements of Image.fromarray
        self.img = self.cvtImg(np.reshape(self.img[0], (128,128)))
        self.img = Image.fromarray(self.img)

        # add image to set of all images (on left of screen)
        self.update_image_bar()

        # set main image to be the one just generated
        self.set_image_element()
    
class LD_Image_Frame(Image_Frame):
    def __init__(self, model, parent_frame, previous_ims, old_im_frames, images):
        super().__init__(model, parent_frame, previous_ims, old_im_frames, images)
        self.printButton.configure(command=partial(self.show_image))
        
    def show_image(self):        
        """
        Show latent diffusion image.
        In this case there are a number of diffusion steps, so a progress bar is helpful if using a slow computer
        """
        self.image_label.configure(image='')
        if self.progressbar != None:
            self.progressbar.destroy()
        self.progressbar = customtkinter.CTkProgressBar(master=self.image_label)        
        self.progressbar.grid(row=1, column=1, padx=(80, 0), pady=(160, 0), sticky="nsew")
        self.progressbar.grid_columnconfigure(0, weight=1)
        self.progressbar.grid_rowconfigure(4, weight=1)
        self.img = np.random.normal(size=(1, 64, 64, 3))
        self.load()
    
    def load(self, i=1):
        # if there are more diffusion steps to do, do them
        if i <= time_steps:
            self.img = self.predict_step(self.img, i-1)
            self.progressbar.set((i*(100/time_steps))/100)
            self.load(i+1)
        # otherwise update images
        else:
            self.img = Image.fromarray(self.cvtImg(self.img)[0])
            self.update_image_bar()
            self.set_image_element()

    def predict_step(self, img, i):
        """
        calls latent diffusion model and outputs a denoised image after 1 denoising step
        """
        img = self.model.predict([img, np.full((1), i)], verbose=0)
        return img

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.width = 695
        self.height = 550
        self.images = []
        self.old_im_frames = []
        self.bg_image = customtkinter.CTkImage(Image.open(os.path.join(absolute_path, "gradient.png")),
                                               size=(self.width, self.height))
        self.bg_image_label = customtkinter.CTkLabel(self, image=self.bg_image)
        self.bg_image_label.place(x=0, y=0)
        self.PIL_img = None
        self.load_models()

        # configure window
        self.title("Light Weight Image Generator")
        self.geometry(f"{self.width}x{self.height}")

        # Create left frame for previous images
        self.left_frame = customtkinter.CTkScrollableFrame(self,
                                                           width=200,
                                                           height=frame_height-93,
                                                           label_text="Previous Images",
                                                           label_fg_color="#2FA572",
                                                           corner_radius=0,
                                                           bg_color="transparent")
        self.left_frame.grid(row=0,  column=0,  padx=10,  pady=5)
        self.left_frame.grid_propagate()
        for i in range(12):
            img_btn= customtkinter.CTkButton(self.left_frame, 
                                    width=90,
                                    height=90,
                                    text="",
                                    corner_radius=0,
                                    fg_color="#333332",
                                    image="")
            img_btn.grid(row=(i//2), column=i%2, padx=5, pady=5)
            self.old_im_frames.append(img_btn)

        # Create right frame for image tabs
        self.right_frame = customtkinter.CTkFrame(self,
                               width=r_frame_width,
                               height=frame_height,
                               corner_radius=0,
                               bg_color="transparent")

        self.right_frame.grid(row=0,  column=1,  padx=10,  pady=5)
                
        # Create image tabs
        self.image_tabs = customtkinter.CTkTabview(self.right_frame, width=250)
        self.image_tabs.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.image_tabs.add("Latent Diffusion Model")
        self.image_tabs.add("Colour GAN")
        self.image_tabs.add("BnW GAN")
        self.image_tabs.grid(row=0, column=0)
        self.LD = LD_Image_Frame(self.model, self.image_tabs.tab("Latent Diffusion Model"), self.left_frame, self.old_im_frames, self.images)
        self.Colour_GAN = Colour_GAN(self.c_gan_model, self.image_tabs.tab("Colour GAN"), self.left_frame, self.old_im_frames, self.images)
        self.BnW_GAN = BnW_GAN(self.bnw_gan_model, self.image_tabs.tab("BnW GAN"), self.left_frame, self.old_im_frames, self.images)

    def load_models(self):        
        """
        Load trained models.
        """
        # replace empty model paths with paths to appropriate models - since large models cant be uploaded to blackboard
        self.model = tf.keras.models.load_model(os.path.join(absolute_path, "")) # empty "" is where model name goes
        self.c_gan_model = tf.keras.models.load_model(os.path.join(absolute_path, ""))
        self.bnw_gan_model = tf.keras.models.load_model(os.path.join(absolute_path, ""))
        self.image = None

if __name__ == "__main__":
    app = App()
    app.mainloop()

    