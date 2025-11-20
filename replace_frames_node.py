import torch
import torch.nn.functional as F

class ReplaceFirstLastFrames:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frames": ("IMAGE",),
                "last_frames": ("IMAGE",),
                "num_start_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "num_last_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "replace_frames"
    CATEGORY = "image/animation"
    
    def replace_frames(self, images, start_frames, last_frames, num_start_frames, num_last_frames):
        # Get dimensions from the main video
        target_height = images.shape[1]
        target_width = images.shape[2]
        target_channels = images.shape[3]
        
        # Resize start_frames and last_frames to match video dimensions
        start_frames = self._resize_frames(start_frames, target_height, target_width, target_channels)
        last_frames = self._resize_frames(last_frames, target_height, target_width, target_channels)
        
        # Get batch sizes
        num_images = images.shape[0]
        num_start_available = start_frames.shape[0]
        num_last_available = last_frames.shape[0]
        
        # Edge case: If no frames to replace, return original
        if num_start_frames <= 0 and num_last_frames <= 0:
            return (images,)
        
        # Clamp the number of frames to replace to reasonable values
        # If the total requested exceeds available frames, adjust proportionally
        if num_start_frames + num_last_frames > num_images:
            if num_images == 1:
                # Special case: only 1 frame total
                # Prioritize start frame
                if num_start_frames > 0:
                    start_to_insert = self._prepare_frames(start_frames, 1)
                    return (start_to_insert,)
                else:
                    last_to_insert = self._prepare_frames(last_frames, 1)
                    return (last_to_insert,)
            
            # Adjust to fit available frames
            total_requested = num_start_frames + num_last_frames
            ratio = num_images / total_requested
            
            # Calculate adjusted values (always keep at least 1 frame in middle if possible)
            adjusted_start = max(0, int(num_start_frames * ratio))
            adjusted_last = max(0, int(num_last_frames * ratio))
            
            # Make sure we don't exceed the total
            while adjusted_start + adjusted_last >= num_images and num_images > 1:
                if adjusted_start > adjusted_last:
                    adjusted_start = max(0, adjusted_start - 1)
                else:
                    adjusted_last = max(0, adjusted_last - 1)
            
            # If still too many, just split evenly
            if adjusted_start + adjusted_last >= num_images:
                adjusted_start = num_images // 2
                adjusted_last = num_images - adjusted_start
            
            num_start_frames = adjusted_start
            num_last_frames = adjusted_last
        
        # Prepare start frames to insert
        start_to_insert = self._prepare_frames(start_frames, num_start_frames) if num_start_frames > 0 else None
        
        # Prepare last frames to insert
        last_to_insert = self._prepare_frames(last_frames, num_last_frames) if num_last_frames > 0 else None
        
        # Build the output sequence
        parts = []
        
        # Add start frames if any
        if start_to_insert is not None:
            parts.append(start_to_insert)
        
        # Add middle frames if any remain
        middle_start_idx = num_start_frames
        middle_end_idx = num_images - num_last_frames
        
        if middle_end_idx > middle_start_idx:
            middle_frames = images[middle_start_idx:middle_end_idx]
            parts.append(middle_frames)
        
        # Add last frames if any
        if last_to_insert is not None:
            parts.append(last_to_insert)
        
        # Concatenate all parts
        if len(parts) == 0:
            # Shouldn't happen, but return original as fallback
            return (images,)
        elif len(parts) == 1:
            output = parts[0]
        else:
            output = torch.cat(parts, dim=0)
        
        return (output,)
    
    def _resize_frames(self, frames, target_height, target_width, target_channels):
        # Check if resize is needed
        if frames.shape[1] == target_height and frames.shape[2] == target_width and frames.shape[3] == target_channels:
            return frames
        
        # ComfyUI format: [batch, height, width, channels]
        # PyTorch interpolate needs: [batch, channels, height, width]
        frames_permuted = frames.permute(0, 3, 1, 2)
        
        # Resize spatial dimensions
        if frames.shape[1] != target_height or frames.shape[2] != target_width:
            frames_permuted = F.interpolate(
                frames_permuted,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Convert back to ComfyUI format
        frames_resized = frames_permuted.permute(0, 2, 3, 1)
        
        # Handle channel mismatch (e.g., RGB vs RGBA)
        if frames_resized.shape[3] != target_channels:
            if target_channels == 3 and frames_resized.shape[3] == 4:
                # RGBA to RGB: drop alpha
                frames_resized = frames_resized[:, :, :, :3]
            elif target_channels == 4 and frames_resized.shape[3] == 3:
                # RGB to RGBA: add alpha channel (fully opaque)
                alpha = torch.ones(frames_resized.shape[0], frames_resized.shape[1], frames_resized.shape[2], 1, 
                                   dtype=frames_resized.dtype, device=frames_resized.device)
                frames_resized = torch.cat([frames_resized, alpha], dim=3)
            elif frames_resized.shape[3] == 1 and target_channels == 3:
                # Grayscale to RGB: repeat channel
                frames_resized = frames_resized.repeat(1, 1, 1, 3)
            elif frames_resized.shape[3] == 1 and target_channels == 4:
                # Grayscale to RGBA: repeat channel + add alpha
                frames_resized = frames_resized.repeat(1, 1, 1, 3)
                alpha = torch.ones(frames_resized.shape[0], frames_resized.shape[1], frames_resized.shape[2], 1,
                                   dtype=frames_resized.dtype, device=frames_resized.device)
                frames_resized = torch.cat([frames_resized, alpha], dim=3)
        
        return frames_resized
    
    def _prepare_frames(self, source_frames, num_needed):
        if num_needed <= 0:
            return None
        
        num_available = source_frames.shape[0]
        
        if num_available == num_needed:
            # Perfect match
            return source_frames
        elif num_available == 1:
            # Repeat single frame
            return source_frames.repeat(num_needed, 1, 1, 1)
        elif num_available > num_needed:
            # More frames than needed, use the first ones
            return source_frames[:num_needed]
        else:
            # Fewer frames than needed, cycle through them
            # Calculate how many full cycles + remaining frames
            full_cycles = num_needed // num_available
            remainder = num_needed % num_available
            
            if remainder == 0:
                # Perfect multiple
                return source_frames.repeat(full_cycles, 1, 1, 1)
            else:
                # Need partial cycle
                full_part = source_frames.repeat(full_cycles, 1, 1, 1)
                partial_part = source_frames[:remainder]
                return torch.cat([full_part, partial_part], dim=0)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ReplaceFirstLastFrames": ReplaceFirstLastFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplaceFirstLastFrames": "Replace First & Last Frames"
}

