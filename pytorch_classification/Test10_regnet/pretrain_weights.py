import requests


download_links = {
    "regnetx_200mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_002-e7e85e5c.pth',
    "regnetx_400mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_004-7d0e9424.pth',
    "regnetx_600mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_006-85ec1baa.pth',
    "regnetx_800mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_008-d8b470eb.pth',
    "regnetx_1.6gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth',
    "regnetx_3.2gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_032-ed0c7f7e.pth',
    "regnetx_4.0gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_040-73c2a654.pth',
    "regnetx_6.4gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_064-29278baa.pth',
    "regnetx_8.0gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_080-7c7fcab1.pth',
    "regnetx_12gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_120-65d5521e.pth',
    "regnetx_16gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth',
    "regnetx_32gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_320-8ea38b93.pth',
    "regnety_200mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_002-e68ca334.pth',
    "regnety_400mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_004-0db870e6.pth',
    "regnety_600mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_006-c67e57ec.pth',
    "regnety_800mf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_008-dc900dbe.pth',
    "regnety_1.6gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth',
    "regnety_3.2gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth',
    "regnety_4.0gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_040-f0d569f9.pth',
    "regnety_6.4gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_064-0a48325c.pth',
    "regnety_8.0gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_080-e7f3eb93.pth',
    "regnety_12gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_120-721ba79a.pth',
    "regnety_16gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_160-d64013cd.pth',
    "regnety_32gf": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_320-ba464b29.pth',
}


def main():
    model_name = "regnetx_400mf"
    print("download weights name: " + model_name)

    if model_name not in download_links.keys():
        raise KeyError("{} not in download_links".format(model_name))

    headers = {"Content-Type": "application/json",
               "Connection": "close",
               "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"}

    save_weights = "./" + model_name + ".pth"

    req = requests.get(url=download_links[model_name],
                       stream=True, headers=headers, timeout=10)
    req.raise_for_status()
    info = int(req.headers["Content-Length"])

    accumulate_data = 0
    with open(save_weights, "wb") as f:
        for data in req.iter_content(2048):
            f.write(data)
            accumulate_data += 2048
            print("\rdownload: [{}Mb/{}Mb] {}%".format(int(accumulate_data / 1024 / 1024),
                                                       int(info / 1024 / 1024),
                                                       int(accumulate_data / info * 100)), end="")
    req.close()


if __name__ == '__main__':
    main()
