from darknet import CNNBlock, Darknet
import torch
import torch.nn as nn

# Darknet 네트워크 구조 설정
architecture_config = [
    (7, 64, 2, 3),  # (kernel size, number of filters, stride, padding)
    "M",  # max-pooling 2x2 stride = 2
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [(conv1), (conv2), repeat times]
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# YOLOv1 모델 정의
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]

                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                num_repeats = x[2]  # integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C+B*5)),  # (S,S,30)
        )

def test():

    split_size = 7  # S (그리드 크기)
    num_boxes = 2   # B (각 셀이 예측하는 바운딩 박스 수)
    num_classes = 20  # C (클래스 수)

    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)

    # 더미 입력 데이터 생성 (batch_size: 1, channel: 3, height: 448, width: 448)
    dummy_input = torch.randn(1, 3, 448, 448)

    # 모델에 데이터 전달 및 출력 확인
    output = model(dummy_input)
    
    expected_shape = (1, split_size * split_size * (num_classes + num_boxes * 5))
    print(f"Expected shape: {expected_shape}")
    print("Test passed!" if output.shape == expected_shape else "Test failed!")


if __name__ == "__main__":
    test()
