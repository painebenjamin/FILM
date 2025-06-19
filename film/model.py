# MIT License
import bisect

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

__all__ = [
    "FILMInterpolator",
]


class FILMInterpolator(torch.nn.Module):
    """
    FiLM: Frame Interpolation for Large Motion

    https://arxiv.org/abs/2202.04901
    Fitsum Reda, Janne Jontkanen, Eric Tabellion, Deqing Sun,
    Caroline Pantofaru, and Brian Cutlass
    Google Research, University of Washington
    ECCV 2022
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        """
        Initializes the FILMInterpolator with the given JIT module.
        """
        super().__init__()
        self.module = module

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "benjamin-paine/taproot-common",
        filename: str = "image-interpolation-film-net.fp16.pt",
        subfolder: str | None = None,
        revision: str | None = None,
        device: str | torch.device | int = "cpu",
    ) -> "FILMInterpolator":
        """
        Loads a pretrained FILM model from a specified repository.
        :param repo_id: The repository ID where the model is hosted.
        :param repo_filename: The filename of the model weights.
        :param device: The device to load the model onto (default is "cpu").
        :return: An instance of FILMInterpolator with the loaded model.
        """
        device = (
            torch.device(device)
            if isinstance(device, str)
            else torch.device(f"cuda:{device}") if isinstance(device, int) else device
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
        module = torch.jit.load(path, map_location=device)
        module.eval()
        module.to(device, dtype=torch.float16)
        return cls(module)

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the module's parameters are located.
        This is useful for ensuring that inputs are on the same device as the model.
        """
        return next(self.module.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the module's parameters.
        This is useful for ensuring that inputs are of the same type as the model.
        """
        return next(self.module.parameters()).dtype

    def pad_image(
        self, image: torch.Tensor, nearest: int = 128
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        Pad the image tensor to ensure it has dimensions divisible by 128.
        :param image: The input image tensor ([C,H,W] or [B,C,H,W]).
        :param nearest: The nearest multiple to pad to (default is 128).
        :return: A tuple containing the padded image and the padding sizes.
        """

        squeeze = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            squeeze = True

        h, w = image.shape[2:]
        pad_h = (nearest - h % nearest) % nearest
        pad_w = (nearest - w % nearest) % nearest
        pad_l = pad_w // 2
        pad_t = pad_h // 2
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t
        padding = (pad_l, pad_r, pad_t, pad_b)
        padded_image = F.pad(image, padding)

        if squeeze:
            padded_image = padded_image.squeeze(0)

        return padded_image, padding

    def unpad_image(
        self, image: torch.Tensor, padding: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Unpad the image tensor to remove the padding.
        :param image: The padded image tensor ([C,H,W] or [B,C,H,W]).
        :param padding: The padding sizes (top, bottom, left, right).
        :return: The unpadded image tensor.
        """
        squeeze = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            squeeze = True

        h, w = image.shape[2:]
        pad_l, pad_r, pad_t, pad_b = padding

        crop_l = pad_l
        crop_r = w - pad_r
        crop_t = pad_t
        crop_b = h - pad_b
        cropped_image = image[:, :, crop_t:crop_b, crop_l:crop_r]

        if squeeze:
            cropped_image = cropped_image.squeeze(0)

        return cropped_image

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare a single tensor by ensuring it has the correct shape and type.
        :param tensor: The input tensor ([C,H,W] or [B,C,H,W]).
        :return: The prepared tensor.
        """
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] == 4:
            tensor = tensor[:, :3, :, :]
        elif tensor.shape[1] != 3:
            raise ValueError("Tensor must have 1, 3, or 4 channels.")

        if tensor.dtype is torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor.to(self.device, dtype=self.dtype)

    def prepare_tensors(
        self,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Prepare multiple tensors by ensuring they have the correct shape and type.
        """
        prepared_tensors = [self.prepare_tensor(tensor) for tensor in tensors]
        first_shape = prepared_tensors[0].shape

        assert all(
            tensor.shape == first_shape for tensor in prepared_tensors
        ), "All tensors must have the same shape."

        return tuple(prepared_tensors)

    def interpolate_video(
        self,
        video: torch.Tensor,
        num_frames: int = 1,
        loop: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        """
        Interpolate frames in a video tensor.
        :param video: The video tensor ([B,C,H,W]).
        :param num_frames: The number of frames to interpolate.
        :param loop: Whether to loop the video.
        :return: A tensor containing the interpolated frames.
        """
        video = self.prepare_tensor(video)
        b, c, h, w = video.shape
        assert b >= 2, "Video must have at least 2 frames for interpolation."

        num_interpolated_frames = b * num_frames
        if loop:
            num_interpolated_frames += 1
        else:
            num_interpolated_frames -= num_frames

        num_output_frames = b + num_interpolated_frames
        results = torch.zeros(
            (num_output_frames, 3, h, w),
            dtype=self.dtype,
            device=video.device,
        )

        iterator = range(b)
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Interpolating", unit="frame", total=b - 1)  # type: ignore[assignment]

        for i in iterator:
            left = video[i]
            if i == b - 1:
                if not loop:
                    break
                right = video[0]
            else:
                right = video[i + 1]

            start_i = i * (num_frames + 1)
            results[start_i] = left

            interpolated_frames = self.forward(
                left,
                right,
                num_frames=num_frames,
                include_start=True,
                include_end=True,
            )
            results[start_i : start_i + num_frames + 2] = interpolated_frames

        if loop:
            results = results[:-1]
        else:
            results[-1] = video[-1]

        return results

    @torch.inference_mode()
    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_frames: int = 1,
        include_start: bool = False,
        include_end: bool = False,
    ) -> torch.Tensor:
        """
        Runs the frame interpolation network, returning all frames including
        the start and end frames.
        """
        start, end = self.prepare_tensors(start, end)

        b, c, h, w = start.shape
        assert b == 1, "Batch size must be 1. For videos, use `.interpolate_video()`."
        start, padding_start = self.pad_image(start)
        end, _ = self.pad_image(end)

        indexes = [0, num_frames + 1]
        remains = list(range(1, num_frames + 1))
        splits = torch.linspace(0, 1, num_frames + 2)
        results = [start, end]

        for i in range(len(remains)):
            starts = splits[indexes[:-1]]
            ends = splits[indexes[1:]]

            distances = (
                (splits[None, remains] - starts[:, None])
                / (ends[:, None] - starts[:, None])
                - 0.5
            ).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)  # type: ignore[arg-type]
            end_i = start_i + 1

            x_0 = results[start_i]
            x_1 = results[end_i]

            d_t = x_0.new_full(
                (1, 1),
                (splits[remains[step]] - splits[indexes[start_i]]),  # type: ignore[arg-type]
            ) / (splits[indexes[end_i]] - splits[indexes[start_i]])

            pred = self.module(x_0, x_1, d_t)
            insert_position = bisect.bisect_left(indexes, remains[step])
            indexes.insert(insert_position, remains[step])
            results.insert(insert_position, pred.clamp(0, 1))

            del remains[step]

        result = torch.cat(results, dim=0)
        result = self.unpad_image(result, padding_start)

        if not include_start:
            result = result[1:]
        if not include_end:
            result = result[:-1]

        return result
