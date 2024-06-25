import os
import torch
from model import create_deep_pose_model


def main():
    img_hw = [256, 256]
    num_keypoints = 98
    weights_path = "./weights/model_weights_209.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = create_deep_pose_model(num_keypoints=num_keypoints)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cpu")["model"])
    model.to(device)

    model.eval()
    with torch.inference_mode():
        x = torch.randn(size=(1, 3, img_hw[0], img_hw[1]), device=device)
        torch.onnx.export(model=model,
                          args=(x,),
                          f="deeppose.onnx")


if __name__ == '__main__':
    main()
