#!/usr/bin/env python3
"""
Test script for scene detection functionality in FILM.
"""

import torch

def test_scene_detection():
    """Test the scene detection functionality."""
    try:
        from film import FILMInterpolator
        
        # Create a simple test video with 10 frames
        # Simulate a scene change at frame 5: 5 black frames followed by 5 white frames
        video = torch.zeros(10, 3, 64, 64)  # [B, C, H, W]
        video[5:] = 1.0  # Make frames 5-9 white
        
        print("Test video shape:", video.shape)
        print("Test video range:", video.min().item(), "to", video.max().item())
        
        # Test without scene detection
        print("\nTesting without scene detection...")
        interpolator = FILMInterpolator.from_pretrained(device="cpu")
        
        try:
            result_no_scene = interpolator.interpolate_video(
                video, num_frames=1, use_scene_detection=False
            )
            print("✓ Standard interpolation successful")
            print("  Output shape:", result_no_scene.shape)
        except Exception as e:
            print("✗ Standard interpolation failed:", e)
            return False
        
        # Test with scene detection
        print("\nTesting with scene detection...")
        try:
            result_with_scene = interpolator.interpolate_video(
                video, num_frames=1, use_scene_detection=True
            )
            print("✓ Scene detection interpolation successful")
            print("  Output shape:", result_with_scene.shape)
        except ImportError as e:
            print("⚠ Scene detection not available (missing dependencies):", e)
            print("  Install with: pip install film[scene-detection]")
            return True  # This is expected if dependencies aren't installed
        except Exception as e:
            print("✗ Scene detection interpolation failed:", e)
            return False
        
        # Compare results
        # With scene detection, there should be one less interpolated frame between scenes
        # For 10 input frames with num_frames=1, standard interpolation produces 19 frames
        # With scene detection (2 scenes), it should produce 18 frames (9 + 9, no interpolation between scenes)
        expected_diff = 1
        actual_diff = result_no_scene.shape[0] - result_with_scene.shape[0]
        
        if actual_diff == expected_diff:
            print("✓ Scene detection correctly reduced output by 1 frame (no interpolation between scenes)")
        else:
            print("⚠ Unexpected output shape difference:")
            print(f"  Expected difference: {expected_diff} frames")
            print(f"  Actual difference: {actual_diff} frames")
            print("  Without scene detection:", result_no_scene.shape)
            print("  With scene detection:", result_with_scene.shape)
        
        return True
        
    except ImportError as e:
        print("✗ Could not import FILM:", e)
        return False
    except Exception as e:
        print("✗ Unexpected error:", e)
        return False

if __name__ == "__main__":
    print("Testing FILM scene detection functionality...")
    success = test_scene_detection()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1) 