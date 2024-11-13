# Notes:
# 0. Built with torch@8a80cee2f3d, torchvision@7d077f1312, ran on Mac M1
#
# 1. I had to change the example input they provided (which had batch_size = 1).
# The stacktrace is more dynamo/export developer facing than user facing imo
# (big stack trace), and the suggested fix doesn't quite help: `num_images = 1`.
#   - Can we improve the error message & action item by automating a bit on this
#     0/1 specialization problem?
#
# 2. Perf:
#  original model took 0.1146 seconds
#  exported model took 0.1152 seconds, speed-up: 0.9944835162402748
#  aoti model took 0.0987 seconds, speed-up: 1.1607539572946162


import torch
import clip
import os
import time
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
original_model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)


def bench(fn):
    # Warm it up
    for i in range(3):
        fn()

    # Time it
    start = time.perf_counter()
    res = fn()
    end = time.perf_counter()
    return res, end - start


with torch.no_grad():
    # Export doesn't like dynamic dimension being 1 in example inputs
    image = torch.cat((image, image))


    # Run the model as is.
    original_output, original_dur = bench(lambda: original_model(image, text))
    print(f"original model took {original_dur:.4f} seconds")


    # Export
    num_images = torch.export.Dim("num_images", min=1, max=1024)
    num_texts = torch.export.Dim("num_texts", min=1, max=1024)
    example_inputs = (image, text)
    dynamic_shapes = { "image": { 0 : num_images }, "text" : { 0 : num_texts } }
    exported = torch.export.export(original_model, example_inputs, dynamic_shapes=dynamic_shapes)
    #print(exported) # visualize the exported model


    # Run exported model
    exported_model = exported.module()
    exported_output, exported_dur = bench(lambda: exported_model(image, text))
    exported_output_0_match = torch.allclose(original_output[0], exported_output[0])
    exported_output_1_match = torch.allclose(original_output[1], exported_output[1])
    print(f"exported model took {exported_dur:.4f} seconds, speed-up: {original_dur / exported_dur}")
    print(f"{exported_output_0_match=}")
    print(f"{exported_output_1_match=}")


    # Run AOTI compile and package
    aoti_output_path = torch._inductor.aoti_compile_and_package(
        exported,
        example_inputs,
        package_path=os.path.join(os.getcwd(), "model.pt2"),
    )


    # Load and run AOTI model
    aoti_model = torch._inductor.aoti_load_package(aoti_output_path)
    aoti_output, aoti_dur = bench(lambda: aoti_model(image, text))
    aoti_output_0_match = torch.allclose(original_output[0], aoti_output[0])
    aoti_output_1_match = torch.allclose(original_output[1], aoti_output[1])
    print(f"aoti model took {aoti_dur:.4f} seconds, speed-up: {original_dur / aoti_dur}")
    print(f"{aoti_output_0_match=}")
    print(f"{aoti_output_1_match=}")