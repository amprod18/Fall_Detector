# Fall Detection System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The Fall Detection System is an image processing project developed as a part of a university project. It utilizes the HOGCV AI algorithm (a human detection system from OpenCV) to identify people in images or videos and then applies various techniques to determine if a person is falling or not. The project includes an evaluation of different image processing approaches to measure speed and precision, such as applying colormaps, multithreading and different filters. Additionally, a comprehensive report presenting the results is available in the repository.

## Features

- Integration of the HOGCV AI algorithm for person detection.
- Implementation of various image processing techniques to assess fall detection performance.
- Comparative analysis of different approaches, including colormaps, multithreading, and filters.
- Inclusion of a detailed report documenting the project's findings.
<p align="center">
  <img src="https://github.com/amprod18/Fall_Detector/blob/main/dataset/test_image1.jpg" alt="input_image" height="200 px" length="200 px"><img src="https://github.com/amprod18/Fall_Detector/blob/main/processed/test_image1_checked.jpg" alt="processed_image" height="200 px" length="200 px">
</p>

## Future Enhancements

- Usage of a better human detector as the HOGCV model is very inconsistent when you play around with light and perspective.
- Speed: Even when using multithreading the program reached around 2-3 fps which would not yield good results in real-time applications.
- Having various threads would convert the script to a time dependant script, which means that coordination and synchronism beetwen threads is a must.

## License

This project is licensed under the [MIT License](LICENSE).
