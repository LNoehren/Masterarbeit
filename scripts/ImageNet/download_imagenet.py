import urllib.request
import codecs
from multiprocessing import Pool

stop_pos = 150000
num_lines = 14197122


def download_image(data):
    try:
        f = open("/home/lennard/Datasets/ImageNet/" + data[0] + ".jpg", "wb")
        f.write(urllib.request.urlopen(data[1], timeout=5).read())
        f.close()
        return True
    except Exception:
        print("Invalid url: {}".format(data[1]))
        return False


name_url_list = []
with codecs.open("/home/lennard/Schreibtisch/fall11_urls.txt", "r", encoding="utf-8", errors="ignore") as reader:
    current_im = 0

    for line in reader:
        current_im += 1
        print(current_im)
        if current_im < stop_pos:
            continue

        name_url_list.append(line.split("\t"))

pool = Pool(8)
results = pool.map(download_image, name_url_list)

successful = 0
error = 0

for result in results:
    stop_pos += 1
    print("Trying to download image {} of {}".format(stop_pos, num_lines))
    if result:
        successful += 1
    else:
        error += 1

print("downloaded {} images".format(successful))
print("{} errors".format(error))

pool.close()
pool.join()






