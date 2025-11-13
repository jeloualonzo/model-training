"""
🔥 AGGRESSIVE AUGMENTATION FOR CAPSTONE DEMO
- Simulates extreme real-world conditions
- Prevents overfitting on clean iPhone XR samples
- Target: 85-90% training accuracy (not 95%+)
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
def augment_signature(img: np.ndarray) -> np.ndarray:
    """
    🔥 AGGRESSIVE augmentation for demo training
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot augment empty image")
    
    original_img = img.copy()
    
    try:
        # STEP 1: ALWAYS geometric (keep as-is)
        img = apply_geometric_transform(img)
        
        if img is None or img.size == 0:
            return original_img
        
        # STEP 2: 🔥 INCREASED writing variations (70% chance, was 60%)
        writing_var = np.random.random()
        
        if writing_var < 0.25:  # 25% - Rushed jitter
            img = apply_rushed_jitter(img)
        elif writing_var < 0.45:  # 20% - Stroke thinning
            img = apply_stroke_thinning(img)
        elif writing_var < 0.60:  # 15% - Stroke dropout
            img = apply_stroke_dropout(img)
        elif writing_var < 0.68:  # 8% - Stroke erosion
            img = apply_stroke_erosion(img)
        elif writing_var < 0.70:  # 2% - Pen pressure
            img = apply_pen_pressure(img)
        # else: 30% - No writing variation
        
        if img is None or img.size == 0:
            return original_img
        
        # STEP 3: 🔥 ALWAYS apply capture condition (was 85% chance)
        condition_type = np.random.random()
        
        if condition_type < 0.40:  # 40% - Lighting
            img = apply_lighting_conditions(img)
        elif condition_type < 0.75:  # 35% - Camera quality
            img = apply_camera_quality(img)
        else:  # 25% - Extreme conditions
            img = apply_extreme_conditions(img)
        
        if img is None or img.size == 0:
            return original_img
        
        # Final protection
        img = protect_signature_darkness(img)
        
        return img
        
    except Exception as e:
        print(f"⚠️ Augmentation exception: {e}, using original")
        return original_img
# ============================================================================
# PROTECTION FUNCTION
# ============================================================================
def protect_signature_darkness(img: np.ndarray, threshold: int = 150, max_brightness: int = 200) -> np.ndarray:
    """
    ORIGINAL VERSION (working well)
    
    Optional: Change max_brightness from 200 → 180 to slightly darken light signatures
    """
    if img is None or img.size == 0:
        return img
    
    try:
        signature_mask = img < threshold
        img = np.where(
            signature_mask,
            np.clip(img, 0, 180),  # ← ONLY CHANGE: 200 → 180
            img
        ).astype(np.uint8)
        return img
    except Exception as e:
        print(f"⚠️ Protection failed: {e}")
        return img
# ============================================================================
# GEOMETRIC TRANSFORMATIONS (KEEP AS-IS)
# ============================================================================
def apply_geometric_transform(img: np.ndarray) -> np.ndarray:
    """Geometric transform - rotation, perspective, crop"""
    h, w = img.shape[:2]
    
    if h <= 0 or w <= 0:
        return img
    
    original_img = img.copy()
    
    try:
        angle = (np.random.random() * 10 - 5)
        skew_x = (np.random.random() * 12 - 6) * np.pi / 180
        skew_y = (np.random.random() * 12 - 6) * np.pi / 180
        
        center = (w / 2, h / 2)
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        skew_amount_x = w * np.tan(skew_x)
        skew_amount_y = h * np.tan(skew_y)
        
        skew_amount_x = np.clip(skew_amount_x, -w * 0.3, w * 0.3)
        skew_amount_y = np.clip(skew_amount_y, -h * 0.3, h * 0.3)
        
        pts2 = np.float32([
            [skew_amount_x, skew_amount_y],
            [w - skew_amount_x, skew_amount_y],
            [skew_amount_x, h - skew_amount_y],
            [w - skew_amount_x, h - skew_amount_y]
        ])
        
        M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
        
        img = cv2.warpAffine(img, M_rotate, (w, h), borderValue=255)
        
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return original_img
        
        img = cv2.warpPerspective(img, M_perspective, (w, h), borderValue=255)
        
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return original_img
        
        if np.random.random() > 0.5:
            crop_amount = np.random.random() * 0.1
            crop_side = np.random.randint(0, 4)
        
            current_h, current_w = img.shape[:2]
            min_size = 100
        
            crop_pixels_h = int(current_h * crop_amount)
            crop_pixels_w = int(current_w * crop_amount)
            
            if crop_pixels_h > current_h - min_size:
                crop_pixels_h = max(0, current_h - min_size)
            if crop_pixels_w > current_w - min_size:
                crop_pixels_w = max(0, current_w - min_size)
        
            if crop_pixels_h < 5 and crop_pixels_w < 5:
                return img
        
            try:
                temp_img = None
                
                if crop_side == 0 and (current_h - crop_pixels_h) >= min_size:
                    temp_img = img[crop_pixels_h:, :]
                elif crop_side == 1 and (current_w - crop_pixels_w) >= min_size:
                    temp_img = img[:, :current_w - crop_pixels_w]
                elif crop_side == 2 and (current_h - crop_pixels_h) >= min_size:
                    temp_img = img[:current_h - crop_pixels_h, :]
                elif crop_side == 3 and (current_w - crop_pixels_w) >= min_size:
                    temp_img = img[:, crop_pixels_w:]
                
                if temp_img is not None and temp_img.size > 0 and temp_img.shape[0] > 0 and temp_img.shape[1] > 0:
                    img = cv2.resize(temp_img, (w, h), interpolation=cv2.INTER_LINEAR)
                    
            except Exception:
                pass
        
        return img
        
    except Exception as e:
        print(f"⚠️ Geometric transform error: {e}, using original")
        return original_img
# ============================================================================
# 🔥 AGGRESSIVE WRITING VARIATIONS
# ============================================================================
def apply_rushed_jitter(img: np.ndarray) -> np.ndarray:
    """
    🔥 INCREASED shakiness: 10-14 (was 8-12)
    """
    if img is None or img.size == 0:
        return img
    
    try:
        alpha = 10 + np.random.random() * 4  # 10-14
        sigma = 2.5 + np.random.random() * 1.5  # 2.5-4
        
        random_state = np.random.RandomState(None)
        shape = img.shape
        
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        
        dx = gaussian_filter(dx, sigma) * alpha
        dy = gaussian_filter(dy, sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1))
        )
        
        deformed = map_coordinates(img, indices, order=1, mode='reflect')
        
        return deformed.reshape(shape).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Rushed jitter error: {e}")
        return img
def apply_stroke_dropout(img: np.ndarray) -> np.ndarray:
    """
    🔥 MORE gaps: 2-4 dropouts, 5-15px (was 1-3, 4-12px)
    """
    if img is None or img.size == 0:
        return img
    
    try:
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        num_dropouts = np.random.randint(2, 5)  # 2-4 gaps
        
        for _ in range(num_dropouts):
            gap_x = int(np.random.random() * w)
            gap_y = int(np.random.random() * h)
            
            gap_width = int(5 + np.random.random() * 10)  # 5-15px
            gap_height = int(5 + np.random.random() * 10)
            
            Y, X = np.ogrid[:h, :w]
            ellipse_mask = (
                ((X - gap_x) / gap_width)**2 + 
                ((Y - gap_y) / gap_height)**2
            ) <= 1
            
            mask[ellipse_mask] = 0
        
        img_float = img.astype(np.float32)
        img_with_gaps = img_float * mask + 255 * (1 - mask)
        
        return np.clip(img_with_gaps, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Stroke dropout error: {e}")
        return img
def apply_stroke_thinning(img: np.ndarray) -> np.ndarray:
    """
    🔥 MORE fading: 65-88% (was 70-90%)
    """
    if img is None or img.size == 0:
        return img
    
    try:
        h, w = img.shape[:2]
        fade_type = np.random.random()
        
        if fade_type < 0.4:
            direction = np.random.randint(0, 2)
            fade_start = 0.94 + np.random.random() * 0.06  # 94-100%
            fade_end = 0.65 + np.random.random() * 0.23     # 65-88%
            
            if direction == 0:
                fade_mask = np.linspace(fade_start, fade_end, w)
                fade_mask = np.tile(fade_mask, (h, 1))
            else:
                fade_mask = np.linspace(fade_start, fade_end, h)
                fade_mask = np.tile(fade_mask[:, np.newaxis], (1, w))
                
        else:
            fade_mask = np.ones((h, w))
            num_patches = np.random.randint(2, 4)
            
            for _ in range(num_patches):
                patch_x = int(np.random.random() * w)
                patch_y = int(np.random.random() * h)
                patch_size = int(30 + np.random.random() * 40)
                
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - patch_x)**2 + (Y - patch_y)**2)
                patch_mask = np.clip(1 - dist / patch_size, 0, 1)
                
                fade_strength = 0.70 + np.random.random() * 0.20  # 70-90%
                fade_mask -= patch_mask * (1 - fade_strength)
                fade_mask = np.clip(fade_mask, 0.65, 1.0)
        
        img_float = img.astype(np.float32)
        img_faded = img_float * fade_mask + 255 * (1 - fade_mask)
        
        return np.clip(img_faded, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Stroke thinning error: {e}")
        return img
def apply_stroke_erosion(img: np.ndarray) -> np.ndarray:
    """Simulate broken strokes"""
    if img is None or img.size == 0:
        return img
    
    try:
        kernel_size = np.random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        inverted = 255 - img
        eroded = cv2.erode(inverted, kernel, iterations=1)
        result = 255 - eroded
        
        return result
        
    except Exception as e:
        print(f"⚠️ Stroke erosion error: {e}")
        return img
def apply_pen_pressure(img: np.ndarray) -> np.ndarray:
    """Simulate varying pen pressure"""
    if img is None or img.size == 0:
        return img
    
    try:
        pen_type = np.random.random()
        
        if pen_type < 0.5:
            kernel = np.array([[0, 1, 0], 
                              [1, 1, 1], 
                              [0, 1, 0]], dtype=np.uint8)
            inverted = 255 - img
            eroded = cv2.erode(inverted, kernel, iterations=1)
            result = 255 - eroded
        else:
            kernel = np.ones((3, 3), dtype=np.uint8)
            inverted = 255 - img
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            result = 255 - dilated
        
        return result
        
    except Exception as e:
        print(f"⚠️ Pen pressure error: {e}")
        return img
# ============================================================================
# 🔥 AGGRESSIVE LIGHTING CONDITIONS
# ============================================================================
def apply_lighting_conditions(img: np.ndarray) -> np.ndarray:
    """
    🔥 STRONGER lighting: 20-50% shadows/glare (was 15-40%)
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot apply lighting to empty image")
    
    h, w = img.shape[:2]
    threshold = 150
    signature_mask = img < threshold
    
    img_float = img.astype(np.float32)
    
    # 🔥 WIDER brightness range: 0.75-1.25 (was 0.80-1.20)
    paper_brightness = 0.75 + np.random.random() * 0.5
    
    img_float = np.where(
        signature_mask,
        np.clip(img_float * 0.96, 0, 180),
        np.clip(img_float * paper_brightness, 0, 255)
    )
    img = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # 🔥 INCREASED shadow probability: 60% (was 50%)
    if np.random.random() > 0.4:
        shadow_type = np.random.random()
        
        if shadow_type < 0.5:
            shadow_dir = np.random.randint(0, 4)
            shadow_strength = 0.20 + np.random.random() * 0.30  # 🔥 20-50% (was 15-40%)
            
            if shadow_dir == 0:
                gradient = np.linspace(shadow_strength, 0, h)[:, np.newaxis]
                gradient = np.repeat(gradient, w, axis=1)
            elif shadow_dir == 1:
                gradient = np.linspace(0, shadow_strength, w)[np.newaxis, :]
                gradient = np.repeat(gradient, h, axis=0)
            elif shadow_dir == 2:
                gradient = np.linspace(0, shadow_strength, h)[:, np.newaxis]
                gradient = np.repeat(gradient, w, axis=1)
            else:
                gradient = np.linspace(shadow_strength, 0, w)[np.newaxis, :]
                gradient = np.repeat(gradient, h, axis=0)
        else:
            shadow_x = int(0.3 * w + np.random.random() * 0.4 * w)
            shadow_y = int(0.3 * h + np.random.random() * 0.4 * h)
            shadow_size = int(70 + np.random.random() * 120)
            shadow_strength = 0.25 + np.random.random() * 0.30  # 🔥 25-55% (was 20-45%)
            
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - shadow_x)**2 + (Y - shadow_y)**2)
            gradient = np.clip(1 - dist / shadow_size, 0, 1) * shadow_strength
        
        img_float = img.astype(np.float32)
        img_float = np.clip(img_float * (1 - gradient), 0, 255)
        img = img_float.astype(np.uint8)
    
    # 🔥 INCREASED glare probability: 40% (was 35%)
    if np.random.random() > 0.6:
        glare_x = int(0.2 * w + np.random.random() * 0.6 * w)
        glare_y = int(0.2 * h + np.random.random() * 0.6 * h)
        glare_size = int(50 + np.random.random() * 100)
        glare_strength = 0.20 + np.random.random() * 0.30  # 🔥 20-50% (was 15-40%)
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - glare_x)**2 + (Y - glare_y)**2)
        glare_mask = np.clip(1 - dist / glare_size, 0, 1) * glare_strength
        
        img_float = img.astype(np.float32)
        brightening = glare_mask * 255
        
        img_float = np.where(
            signature_mask,
            img_float,
            np.clip(img_float + brightening, 0, 255)
        )
        img = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # Final protection
    img = np.where(
        signature_mask,
        np.clip(img, 0, 200),
        img
    ).astype(np.uint8)
    
    return img
# ============================================================================
# 🔥 AGGRESSIVE CAMERA QUALITY
# ============================================================================
def apply_camera_quality(img: np.ndarray) -> np.ndarray:
    """
    🔥 STRONGER camera effects
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot apply camera quality to empty image")
    
    quality_type = np.random.random()
    
    if quality_type < 0.30:
        # 🔥 MORE blur: 1-4px (was 1-3px)
        blur_amount = int(1 + np.random.random() * 3)
        if blur_amount % 2 == 0:
            blur_amount += 1
        img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        
    elif quality_type < 0.60:
        # 🔥 MORE noise: 8-18 (was 6-15)
        noise_level = 8 + np.random.random() * 10
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    elif quality_type < 0.80:
        # 🔥 MORE compression: 15% (was 12%)
        block_size = 8
        h, w = img.shape[:2]
        compression_strength = 0.15
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                block = img[y:y_end, x:x_end]
                if block.size > 0:
                    avg_val = np.mean(block)
                    img[y:y_end, x:x_end] = (
                        block * (1 - compression_strength) + 
                        avg_val * compression_strength
                    ).astype(np.uint8)
    
    else:
        # 🔥 LOWER quality: 70% (was 75%)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, enc_img = cv2.imencode('.jpg', img, encode_param)
        dec_img = cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)
        
        if dec_img is not None:
            img = dec_img
    
    return img
# ============================================================================
# 🔥 AGGRESSIVE EXTREME CONDITIONS
# ============================================================================
def apply_extreme_conditions(img: np.ndarray) -> np.ndarray:
    """
    🔥 MORE extreme: ±30% (was ±25%)
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot apply extreme conditions to empty image")
    
    threshold = 150
    signature_mask = img < threshold
    
    extreme_type = np.random.random()
    
    if extreme_type < 0.4:
        # 🔥 DARKER: 70% (was 75%)
        img = np.clip(img * 0.70, 0, 255).astype(np.uint8)
        img = np.clip(128 + 1.3 * (img - 128), 0, 255).astype(np.uint8)
        
    elif extreme_type < 0.7:
        # 🔥 BRIGHTER: 1.30x (was 1.25x)
        img_float = img.astype(np.float32)
        
        img_float = np.where(
            signature_mask,
            img_float * 0.94,
            np.clip(img_float * 1.30, 0, 255)
        )
        
        img_float = 128 + 0.75 * (img_float - 128)
        img = np.clip(img_float, 0, 255).astype(np.uint8)
        
    else:
        # 🔥 MORE blur: 3-7px (was 3-6px)
        kernel_size = int(3 + np.random.random() * 4)
        angle = np.random.random() * 360
        
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        img = cv2.filter2D(img, -1, kernel)
    
    # 🔥 MORE paper variations: 50% chance (was 40%)
    if np.random.random() > 0.5:
        paper_type = np.random.randint(0, 5)
        
        paper_colors = {
            0: 1.0,
            1: 0.97,   # 🔥 MORE cream (was 0.98)
            2: 0.95,   # 🔥 MORE gray (was 0.96)
            3: 0.98,   # 🔥 MORE yellow (was 0.99)
            4: 0.96,   # 🔥 MORE blue (was 0.97)
        }
        
        paper_mult = paper_colors[paper_type]
        
        img_float = img.astype(np.float32)
        img_float = np.where(
            signature_mask,
            img_float,
            np.clip(img_float * paper_mult, 0, 255)
        )
        img = img_float.astype(np.uint8)
    
    # Final protection
    img = np.where(
        signature_mask,
        np.clip(img, 0, 200),
        img
    ).astype(np.uint8)
    
    return img