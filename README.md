# PanoVerseStereo
This repository provides a work-in-progress, demo implementation of a novel deep synthesis and exploration of omnidirectional stereoscopic environments method.

The work aims to introduce an innovative approach to automatically generate and explore immersive stereoscopic indoor environments derived from a single monoscopic panoramic image in an equirectangular format. Once per 360 shot, we estimate the per-pixel depth using a gated deep network architecture. Subsequently, we synthesize a collection of panoramic slices through reprojection and view-synthesis employing deep learning. These slices are distributed around the central viewpoint, with each slice’s projection center placed on the circular path covered by the eyes during a head rotation. Furthermore, each slice encompasses an angular extent sufficient to accommodate the potential gaze directions of both the left and right eye and to provide context for reconstruction. For fast display, a stereoscopic multiple-center-of-projection stereo pair in equirectangular format is composed by suitably blending the precomputed slices. At run-time, the pair is loaded in a lightweight WebXR viewer that responds to head rotations, offering both motion and stereo cues. The approach combines and extends state-of-the-art data-driven techniques, incorporating several innovations. Notably, a gated architecture is introduced for panoramic monocular depth estimation. Leveraging the predicted depth, the same gated architecture is then applied to the re-projection of visible pixels, facilitating the inpainting of occluded and disoccluded regions by incorporating a mixed Generative Adversarial Network (GAN). The resulting system works on a variety of available VR headsets and can serve as a base component for a variety of immersive applications. We will demonstrate our technology on several indoor scenes from publicly available data.

**Method Pipeline overview**:
![](assets/visual-overview.jpg)

## Updates
* 2024-1-11: First release including omnidirectional stereo generation script.
  
## Python Requirements
See file `requirements.txt`
 
## Installation
The installation involves creating a Python virtual environment and installing all the essential Python modules using pip. After cloning the repository, run:

```
# python -m venv .env
# source .env/bin/activate
# pip install -r requirements.txt
```

## Usage
There are three python scripts:
* `panoverse_strip.py` - Synthesize a collection of panoramic slices through reprojection and view-synthesis employing deep learning
* `img2equi.py` - Compose an omnidirectional stereoscopic image pair in equirectangular format by suitably blending the precomputed slices and used for display in a lightweight WebXR viewer
* `aizoom.py` - Upsample the omnidirectional stereoscopic image pair via super-resolution generative adversarial networks

In the `data` directory, you will find the test file named `scene_02082_299.png`, which is an equirectangular image of an indoor environment.

To create a 4K omnidirectional stereoscopic image pair from the original 1024x512 equirectangular scene, run:

```
# python panoverse_strip.py -o tmp --samples 360 --padded data/scene_02082_299.png
# python img2equi.py -o out tmp/scene_02082_299.json
# python aizoom.py --zoom-factor 4 -o out/4Xscene_02082_299_l.png out/scene_02082_299_l.png
# python aizoom.py --zoom-factor 4 -o out/4Xscene_02082_299_r.png out/scene_02082_299_r.png
```

## Acknowledgements
We acknowledge the support of the PNRR ICSC National Research Centre for High Performance Computing, Big Data and Quantum Computing (CN00000013), under the NRRP MUR program funded by the NextGenerationEU.

