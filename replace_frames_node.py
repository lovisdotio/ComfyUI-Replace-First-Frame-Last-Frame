import torch

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

