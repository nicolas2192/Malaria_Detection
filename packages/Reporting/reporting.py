import matplotlib.pyplot as plt
import os


def pdf_generation(img, result, image_path):
    """
    Saves analysed cell in a pdf
    :param img: Image array that was previously analysed
    :param result: Prediction result, Parasitized or Uninfected
    :param image_path: Image path and name, used to name the pdf file
    :return: PDF file with the analysed image
    """
    pdf_path = "data/predictions"
    cnt = len(os.listdir(pdf_path)) + 1
    name = (image_path[::-1][:image_path[::-1].index("/")])[::-1].rstrip(".png")
    name = f"{name}-{cnt}"

    # Matplotlib and pdf generation
    fig, ax = plt.subplots()
    label_font = {"fontname": "Arial", "fontsize": 12}
    img_plot = plt.imshow(img)
    fig.suptitle(result, fontsize=18)
    ax.set_title(image_path, fontdict=label_font)
    plt.savefig(f"data/predictions/{name}.pdf")
    print(f"Image saved as a pdf at {pdf_path}/{name}.pdf")
