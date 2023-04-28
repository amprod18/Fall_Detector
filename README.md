# Fall Detector

As the title suggest its a program which analizes an image in different manners and detects when a person has fallen.

This script uses HOGCV (a human detection system from OpenCV) to detect humans in the images (or videos), then uses the boundary information to calculate diferent properties of the person and decide if it has fallen.

As a fall detector is pretty bad right now... The HOGCV model is very inconsistent when you play around with light and perspective and very slow for real-time applications. So the script is equiped not for real-time scanning but for image scanning experimentation. It has several function to change the colormap used or analyse the image after a filter (like edge detecting filter, aka Prewitt) has been applied. This way one can analyze the performance of each colormap and filter to determine if it is better to use some of this filters below.
The script also has a function to test 22 colormaps sequentially or threadedly to see if multiple threads result in better performance. In my case, using 4 threads gained around 25%-30% in time which might have future in real-time applications. Also havibng various threads would convert the script to a time dependant script and it means that coordination and synchronism beetwen threads is a must.
