# Depth Map to Textured Mesh Generator

![visualization](teaser.gif)

This project contains a 'Textured Mesh Generator from Depth Map' where the depth map is generated by [Marigold depth generation](https://marigoldmonodepth.github.io/) or [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2). 
It takes an input image and a corresponding depth map, which Marigold/DepthAnything generates. 

### Features

- **Image Input**: Accepts standard image formats like JPG, JPEG, and PNG.
- **Depth Map**: npy file generated by [MARIGOLD DIFFUSION](https://marigoldmonodepth.github.io/) / [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2) (or GT depth map converted to numpy array is OK).
    - You can use huggingface demo of the marigold [here](https://huggingface.co/spaces/prs-eth/marigold-lcm)
    - You can use huggingface demo of the depthanything [here](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2)

- **Textured Mesh Output**: Produces a textured mesh that combines the visual information from the image with the structural details from the depth map.

## Usage
Run the processing script to generate the textured mesh.

```bash
    python depth_to_textured_mesh.py --depth <path to depth map npy file> --image <path to input image file>
```

## Acknowledgements

You can also read my blog post for the project: [Post Link](https://velog.io/@gjghks950/Diffusion-%EC%B6%94%EC%A0%95%ED%95%9C-Depth-Map-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%B4-Textured-Mesh-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0-feat.-Marigold)
