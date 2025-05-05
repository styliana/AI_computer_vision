import cv2
import numpy as np
import gradio as gr

#variable that contains the clicked corners
src_points =[]

#function called when we click on th einput image
def on_select(value,evt: gr.EventData): #varialbe ge.EventData required by gradio 
    if len(src_points) < 4:
        src_points.append(evt._data['index'])

    return len(src_points)
#function that fixes the input image
def fiximg(img):
    dst_points = np.float32([[0, 0], [0,800], [600, 800], [600,0]]) #dst points - corners of the img that we want

    # Get the homography matrix
    src_float = np.float32(src_points)
    H = cv2.getPerspectiveTransform(src_float, dst_points)

    # Apply the perspective transformation
    output_img = cv2.warpPerspective(img, H, (600, 800))

    #return the fixed image
    return output_img


#build the gui with blocks
with gr.Blocks() as demo:
    gr.Markdown('Document scanner')
    coord_num = gr.Textbox(label='Number of clicked coordinates: ', value=0)

    #first row contains the imputs
    with gr.Row():
        #upload the ig to be fixed
        inp = gr.Image(label='Input image')
        #event used to get the corners
        inp.select(fn=on_select,inputs=inp,outputs=coord_num)
        #show the fixed image
        out = gr.Image(label='Output image')

    #second row contains the buttons
    with gr.Row():
        btn = gr.Button('Fix!')
        btn.click(fn=fiximg, inputs=inp, outputs=out)
#launch the interface
demo.launch()

'''
# How gradio works very nice
#define the function foo
def foo(name, intensity):
    return 'Hello, ' + name + '!' * int(intensity)

#define a gradio interface
demo = gr.Interface(
    fn = foo,
    inputs = ['text','slider'], #for the argument sof the function
    outputs = ['text'] #what the function foo returns
)

#launch the interface
demo.launch()
'''