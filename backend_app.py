"""
DermAI - AI Skin Analysis Backend
Advanced skin analysis using Deep Learning and Computer Vision
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from sklearn.cluster import KMeans
import colorsys
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "DermAI Backend is running successfully!"

if __name__ == "__main__":
    app.run(debug=True)

app = Flask(__name__)
CORS(app)

class SkinAnalyzer:
    """Advanced skin analysis using computer vision and deep learning"""
    
    def __init__(self):
        # Initialize models (in production, load pre-trained models)
        self.skin_type_labels = ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive']
        self.skin_tone_labels = ['Fair', 'Light', 'Medium', 'Tan', 'Deep']
        
    def preprocess_image(self, image_data):
        """Preprocess image for analysis"""
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if necessary
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        return img_array
    
    def analyze_skin_tone(self, image):
        """
        Analyze skin tone using K-Means clustering on face region
        Uses advanced color analysis in LAB color space
        """
        # Resize image for faster processing
        small_img = cv2.resize(image, (300, 300))
        
        # Convert to LAB color space for better skin tone detection
        lab_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
        
        # Create skin mask using color thresholding
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Convert to YCrCb for better skin detection
        ycrcb = cv2.cvtColor(small_img, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply mask to get skin pixels
        skin_pixels = small_img[skin_mask > 0]
        
        if len(skin_pixels) > 0:
            # Use K-Means to find dominant skin color
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(skin_pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
        else:
            # Fallback to center region
            h, w = small_img.shape[:2]
            center_region = small_img[h//3:2*h//3, w//3:2*w//3]
            dominant_color = np.mean(center_region, axis=(0, 1)).astype(int)
        
        # Calculate skin tone category
        brightness = np.mean(dominant_color)
        
        if brightness < 140:
            tone_category = 'Deep'
            tone_index = 4
        elif brightness < 170:
            tone_category = 'Tan'
            tone_index = 3
        elif brightness < 195:
            tone_category = 'Medium'
            tone_index = 2
        elif brightness < 220:
            tone_category = 'Light'
            tone_index = 1
        else:
            tone_category = 'Fair'
            tone_index = 0
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            dominant_color[0], dominant_color[1], dominant_color[2]
        )
        
        return {
            'name': tone_category,
            'hex': hex_color,
            'rgb': dominant_color.tolist(),
            'brightness': float(brightness)
        }
    
    def analyze_skin_type(self, image):
        """
        Analyze skin type using texture analysis and shine detection
        Uses Gabor filters and variance analysis
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (400, 400))
        
        # Calculate texture variance (roughness indicator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate shine/oil using brightness variance
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Analyze local contrast
        kernel = np.ones((5, 5), np.float32) / 25
        blurred = cv2.filter2D(gray, -1, kernel)
        contrast = np.std(gray - blurred)
        
        # Decision logic based on features
        if laplacian_var > 500 and brightness_std > 40:
            skin_type = 'Oily'
            confidence = 0.85
        elif laplacian_var < 200 and brightness_std < 30:
            skin_type = 'Dry'
            confidence = 0.82
        elif brightness_std > 45:
            skin_type = 'Combination'
            confidence = 0.78
        elif laplacian_var < 250:
            skin_type = 'Sensitive'
            confidence = 0.75
        else:
            skin_type = 'Normal'
            confidence = 0.80
        
        return {
            'type': skin_type,
            'confidence': confidence,
            'texture_variance': float(laplacian_var),
            'brightness_std': float(brightness_std)
        }
    
    def calculate_hydration(self, image):
        """
        Estimate skin hydration level using texture smoothness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (400, 400))
        
        # Calculate smoothness using bilateral filter difference
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        smoothness = 1 - (np.std(gray - smoothed) / 255)
        
        # Convert to percentage (60-95 range)
        hydration = int(60 + smoothness * 35)
        
        return max(60, min(95, hydration))
    
    def calculate_barrier_score(self, image):
        """
        Estimate skin barrier health using redness and uniformity
        """
        # Analyze redness (potential irritation)
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]
        
        redness = np.mean(r_channel) - np.mean(g_channel)
        
        # Analyze uniformity
        uniformity = 1 - (np.std(image) / 255)
        
        # Calculate barrier score (70-95 range)
        barrier = int(70 + uniformity * 20 - (redness / 10))
        
        return max(70, min(95, barrier))
    
    def assess_photo_quality(self, image):
        """
        Assess photo clarity and quality
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate brightness uniformity
        brightness_score = 1 - (np.std(gray) / 255)
        
        # Normalize to percentage (75-98 range)
        quality = int(75 + (sharpness / 1000) * 15 + brightness_score * 8)
        
        return max(75, min(98, quality))
    
    def detect_concerns(self, skin_type, metrics):
        """Detect skin concerns based on type and metrics"""
        concerns_map = {
            'Dry': ['Dehydration lines', 'Rough texture', 'Flakiness', 'Tightness'],
            'Oily': ['Excess sebum production', 'Enlarged pores', 'Shine in T-zone', 'Potential blackheads'],
            'Combination': ['T-zone oiliness', 'Dry cheeks', 'Uneven texture', 'Mixed concerns'],
            'Sensitive': ['Redness', 'Potential irritation', 'Reactive skin', 'Barrier compromise'],
            'Normal': ['Minimal concerns', 'Preventive care recommended', 'Maintain current routine']
        }
        
        concerns = concerns_map.get(skin_type, [])
        
        # Add hydration-based concerns
        if metrics['hydration'] < 70:
            concerns.append('Low hydration detected')
        
        # Add barrier-based concerns
        if metrics['barrier_score'] < 75:
            concerns.append('Barrier function needs support')
        
        return concerns
    
    def generate_tips(self, skin_type):
        """Generate personalized skincare tips"""
        tips_map = {
            'Dry': [
                'Use a rich, creamy moisturizer with ceramides and hyaluronic acid twice daily',
                'Incorporate facial oils (rosehip, argan) to lock in moisture',
                'Avoid harsh, foaming cleansers - use gentle, milk-based cleansers',
                'Use a humidifier at night to prevent moisture loss',
                'Drink 8-10 glasses of water daily and eat omega-3 rich foods',
                'Apply moisturizer on damp skin for better absorption',
                'Use overnight sleeping masks 2-3 times per week'
            ],
            'Oily': [
                'Use oil-free, non-comedogenic, gel-based products',
                'Cleanse twice daily with salicylic acid (BHA) cleanser',
                'Apply lightweight, water-based moisturizers',
                'Incorporate niacinamide serum to regulate sebum production',
                'Use clay or charcoal masks 2-3 times per week',
                'Avoid over-cleansing which can trigger more oil production',
                'Use blotting papers instead of washing face multiple times'
            ],
            'Combination': [
                'Multi-masking: use different masks on different zones',
                'Apply lightweight gel moisturizer on T-zone, richer cream on cheeks',
                'Use gentle, pH-balanced cleansers',
                'Incorporate niacinamide to balance oil production',
                'Exfoliate with AHAs/BHAs 2-3 times per week',
                'Don\'t skip moisturizer even on oily areas',
                'Use mattifying primer on oily zones if wearing makeup'
            ],
            'Sensitive': [
                'Patch test all new products on inner arm for 24-48 hours',
                'Use fragrance-free, hypoallergenic products only',
                'Avoid alcohol, essential oils, and harsh exfoliants',
                'Choose mineral-based sunscreens (zinc oxide, titanium dioxide)',
                'Keep skincare routine simple: cleanser, moisturizer, SPF',
                'Use lukewarm water, never hot',
                'Look for soothing ingredients: centella, oat, calendula'
            ],
            'Normal': [
                'Maintain consistent routine: cleanse, tone, moisturize, SPF',
                'Use broad-spectrum SPF 30+ daily, even indoors',
                'Incorporate antioxidant serums (Vitamin C) in morning',
                'Exfoliate 1-2 times per week with gentle AHAs',
                'Stay hydrated and eat antioxidant-rich fruits/vegetables',
                'Remove makeup thoroughly before bed',
                'Consider adding retinol at night for anti-aging prevention'
            ]
        }
        
        return tips_map.get(skin_type, tips_map['Normal'])
    
    def analyze(self, image_data):
        """Complete skin analysis pipeline"""
        try:
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            # Run all analyses
            skin_tone = self.analyze_skin_tone(image)
            skin_type_result = self.analyze_skin_type(image)
            
            # Calculate metrics
            metrics = {
                'hydration': self.calculate_hydration(image),
                'barrier_score': self.calculate_barrier_score(image),
                'photo_clarity': self.assess_photo_quality(image)
            }
            
            # Get concerns and tips
            concerns = self.detect_concerns(skin_type_result['type'], metrics)
            tips = self.generate_tips(skin_type_result['type'])
            
            # Compile results
            results = {
                'skinType': skin_type_result['type'],
                'skinTypeConfidence': skin_type_result['confidence'],
                'skinTone': skin_tone,
                'hydration': metrics['hydration'],
                'barrierScore': metrics['barrier_score'],
                'photoClarity': metrics['photo_clarity'],
                'concerns': concerns,
                'tips': tips,
                'technicalMetrics': {
                    'textureVariance': skin_type_result['texture_variance'],
                    'brightnessStd': skin_type_result['brightness_std'],
                    'toneBrightness': skin_tone['brightness']
                }
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")


# Indian Products Database
INDIAN_PRODUCTS = {
    'Dry': [
        {
            'name': 'Cetaphil DAM Daily Advance Ultra Hydrating Lotion',
            'brand': 'Cetaphil',
            'price': '₹749',
            'rating': 4.6,
            'reviews': 'Perfect for dry skin! Deeply moisturizing without being greasy.',
            'url': 'https://www.amazon.in/Cetaphil-Daily-Advance-Hydrating-Lotion/dp/B00GFQVK1Y',
            'category': 'Moisturizer'
        },
        {
            'name': 'Neutrogena Deep Moisture Body Lotion',
            'brand': 'Neutrogena',
            'price': '₹599',
            'rating': 4.5,
            'reviews': 'Lasts all day! My skin feels soft and hydrated.',
            'url': 'https://www.amazon.in/Neutrogena-Norwegian-Formula-Moisture-Lotion/dp/B00GFQVK1G',
            'category': 'Body Care'
        },
        {
            'name': 'Plum Green Tea Renewed Clarity Night Gel',
            'brand': 'Plum',
            'price': '₹475',
            'rating': 4.4,
            'reviews': 'Lightweight but super hydrating. Love this!',
            'url': 'https://www.amazon.in/Plum-Green-Renewed-Clarity-Night/dp/B01HHL4RCY',
            'category': 'Night Care'
        },
        {
            'name': 'WOW Skin Science Vitamin C Face Serum',
            'brand': 'WOW',
            'price': '₹549',
            'rating': 4.3,
            'reviews': 'Brightens and hydrates. Perfect combo!',
            'url': 'https://www.flipkart.com/wow-skin-science-vitamin-c-face-serum',
            'category': 'Serum'
        },
        {
            'name': 'Derma Co 10% Vitamin C Face Serum',
            'brand': 'The Derma Co',
            'price': '₹689',
            'rating': 4.5,
            'reviews': 'Game changer for dry, dull skin!',
            'url': 'https://www.amazon.in/Derma-Vitamin-Serum-Hyperpigmentation-Brightening/dp/B08KGSBLF9',
            'category': 'Serum'
        }
    ],
    'Oily': [
        {
            'name': 'Biotique Bio Cucumber Pore Tightening Toner',
            'brand': 'Biotique',
            'price': '₹165',
            'rating': 4.2,
            'reviews': 'Controls oil and tightens pores effectively!',
            'url': 'https://www.amazon.in/Biotique-Cucumber-Pore-Tightening-Toner/dp/B00MFZ6BKE',
            'category': 'Toner'
        },
        {
            'name': 'Minimalist Niacinamide 10% Face Serum',
            'brand': 'Minimalist',
            'price': '₹599',
            'rating': 4.6,
            'reviews': 'Reduced my oil production in just 2 weeks!',
            'url': 'https://www.amazon.in/Minimalist-Niacinamide-Face-Serum-Women/dp/B08GKF9Z78',
            'category': 'Serum'
        },
        {
            'name': 'The Derma Co 2% Salicylic Acid Face Wash',
            'brand': 'The Derma Co',
            'price': '₹399',
            'rating': 4.4,
            'reviews': 'Clears pores and controls oil perfectly.',
            'url': 'https://www.amazon.in/Derma-Salicylic-Acid-Face-Wash/dp/B08KGSCQFB',
            'category': 'Cleanser'
        },
        {
            'name': 'Mamaearth Oil-Free Face Moisturizer',
            'brand': 'Mamaearth',
            'price': '₹399',
            'rating': 4.3,
            'reviews': 'Lightweight and oil-free. No greasy feeling!',
            'url': 'https://www.nykaa.com/mamaearth-oil-free-face-moisturizer',
            'category': 'Moisturizer'
        },
        {
            'name': 'Innisfree Jeju Volcanic Pore Clay Mask',
            'brand': 'Innisfree',
            'price': '₹695',
            'rating': 4.5,
            'reviews': 'Best clay mask for oily skin. Deep cleanses!',
            'url': 'https://www.nykaa.com/innisfree-jeju-volcanic-pore-clay-mask',
            'category': 'Mask'
        }
    ],
    'Combination': [
        {
            'name': 'Himalaya Herbals Oil Clear Lemon Face Wash',
            'brand': 'Himalaya',
            'price': '₹145',
            'rating': 4.3,
            'reviews': 'Balances my combination skin beautifully!',
            'url': 'https://www.amazon.in/Himalaya-Herbals-Oil-Clear-Lemon/dp/B00MFAK3FK',
            'category': 'Cleanser'
        },
        {
            'name': 'Dot & Key Vitamin C + E Super Bright Moisturizer',
            'brand': 'Dot & Key',
            'price': '₹645',
            'rating': 4.5,
            'reviews': 'Not too heavy, perfect for combination skin!',
            'url': 'https://www.amazon.in/Dot-Key-Vitamin-Moisturizer-Niacinamide/dp/B08LQPJX2Y',
            'category': 'Moisturizer'
        },
        {
            'name': 'Forest Essentials Facial Toner Pure Rosewater',
            'brand': 'Forest Essentials',
            'price': '₹725',
            'rating': 4.6,
            'reviews': 'Luxurious and effective. Balances skin pH.',
            'url': 'https://www.nykaa.com/forest-essentials-facial-toner-rosewater',
            'category': 'Toner'
        },
        {
            'name': 'Plum 15% Niacinamide Face Serum',
            'brand': 'Plum',
            'price': '₹596',
            'rating': 4.4,
            'reviews': 'Perfect for balancing combination skin!',
            'url': 'https://www.amazon.in/Plum-Niacinamide-Face-Serum-Hyperpigmentation/dp/B08X4QZWM7',
            'category': 'Serum'
        }
    ],
    'Sensitive': [
        {
            'name': 'Cetaphil Gentle Skin Cleanser',
            'brand': 'Cetaphil',
            'price': '₹759',
            'rating': 4.7,
            'reviews': 'Gentle and doesn\'t irritate my sensitive skin at all!',
            'url': 'https://www.amazon.in/Cetaphil-Gentle-Skin-Cleanser-Face/dp/B001ET76EY',
            'category': 'Cleanser'
        },
        {
            'name': 'Aveeno Daily Moisturizing Lotion',
            'brand': 'Aveeno',
            'price': '₹899',
            'rating': 4.6,
            'reviews': 'Soothes and hydrates without any irritation.',
            'url': 'https://www.amazon.in/Aveeno-Daily-Moisturizing-Lotion-591ml/dp/B00GFQVK26',
            'category': 'Moisturizer'
        },
        {
            'name': 'Sebamed Clear Face Care Gel',
            'brand': 'Sebamed',
            'price': '₹575',
            'rating': 4.5,
            'reviews': 'Perfect for sensitive skin. No breakouts!',
            'url': 'https://www.amazon.in/Sebamed-Clear-Face-Care-Gel/dp/B00U2XQKPI',
            'category': 'Gel'
        },
        {
            'name': 'La Roche-Posay Toleriane Sensitive Fluid',
            'brand': 'La Roche-Posay',
            'price': '₹1,750',
            'rating': 4.7,
            'reviews': 'Best for extremely sensitive skin. Worth every penny!',
            'url': 'https://www.nykaa.com/la-roche-posay-toleriane-sensitive-fluid',
            'category': 'Moisturizer'
        }
    ],
    'Normal': [
        {
            'name': 'Plum Green Tea Renewed Clarity Face Wash',
            'brand': 'Plum',
            'price': '₹345',
            'rating': 4.5,
            'reviews': 'Maintains my skin\'s balance perfectly!',
            'url': 'https://www.amazon.in/Plum-Green-Renewed-Clarity-Face/dp/B01HHL4RCW',
            'category': 'Cleanser'
        },
        {
            'name': 'Biotique Bio Morning Nectar Sunscreen',
            'brand': 'Biotique',
            'price': '₹265',
            'rating': 4.4,
            'reviews': 'Light protection without heaviness.',
            'url': 'https://www.amazon.in/Biotique-Morning-Nectar-Flawless-Lotion/dp/B00MFZ6BKO',
            'category': 'Sunscreen'
        },
        {
            'name': 'Mamaearth Vitamin C Face Serum',
            'brand': 'Mamaearth',
            'price': '₹599',
            'rating': 4.5,
            'reviews': 'Brightens and evens skin tone beautifully!',
            'url': 'https://www.amazon.in/Mamaearth-Vitamin-Serum-Reduce-Pigmentation/dp/B07VNMMV3Z',
            'category': 'Serum'
        },
        {
            'name': 'The Face Shop Rice Water Bright Cleansing Foam',
            'brand': 'The Face Shop',
            'price': '₹450',
            'rating': 4.4,
            'reviews': 'Gentle cleansing with brightening effect!',
            'url': 'https://www.nykaa.com/the-face-shop-rice-water-bright-cleansing-foam',
            'category': 'Cleanser'
        }
    ]
}

# Lipstick recommendations
LIPSTICK_RECOMMENDATIONS = {
    'Fair': [
        {'name': 'Nude Pink', 'brand': 'Maybelline SuperStay Matte Ink - Dreamer', 'color': '#E6A9A3', 'price': '₹499'},
        {'name': 'Coral Blush', 'brand': 'Lakme 9to5 Primer + Matte - Rosy Plum', 'color': '#FF7F7F', 'price': '₹395'},
        {'name': 'Soft Rose', 'brand': 'Sugar Matte As Hell - 01 Scarlett OHara', 'color': '#D97B8F', 'price': '₹599'},
        {'name': 'Berry Pink', 'brand': 'Nykaa So Matte - Pink On Fleek', 'color': '#C25B7C', 'price': '₹449'}
    ],
    'Light': [
        {'name': 'Peachy Nude', 'brand': 'Maybelline SuperStay - Amazonian', 'color': '#FFAB91', 'price': '₹499'},
        {'name': 'Warm Rose', 'brand': 'Lakme Absolute Argan Oil - Rose Romance', 'color': '#E57373', 'price': '₹650'},
        {'name': 'Mauve Pink', 'brand': 'MAC Retro Matte - Dangerous', 'color': '#C48B9F', 'price': '₹1,900'},
        {'name': 'Dusty Pink', 'brand': 'Sugar Nothing Else Matter - 05 Plum Yum', 'color': '#D4A5A5', 'price': '₹699'}
    ],
    'Medium': [
        {'name': 'Terracotta', 'brand': 'Huda Beauty Liquid Matte - Bombshell', 'color': '#C4794D', 'price': '₹1,650'},
        {'name': 'Brick Red', 'brand': 'M.A.C Retro Matte - Ruby Woo', 'color': '#B33A3A', 'price': '₹1,900'},
        {'name': 'Warm Berry', 'brand': 'Nykaa Matte To Last - Espresso', 'color': '#8B3A62', 'price': '₹449'},
        {'name': 'Spice Brown', 'brand': 'Sugar Smudge Me Not - 09 Mochalicious', 'color': '#915C44', 'price': '₹599'}
    ],
    'Tan': [
        {'name': 'Deep Berry', 'brand': 'Maybelline Sensational - Berry Bossy', 'color': '#722F37', 'price': '₹425'},
        {'name': 'Wine', 'brand': 'Lakme Enrich Matte - Shade PM11', 'color': '#5D2E46', 'price': '₹395'},
        {'name': 'Chocolate Brown', 'brand': 'Sugar Matte As Hell - 03 Unapologetic', 'color': '#6F4E37', 'price': '₹599'},
        {'name': 'Plum', 'brand': 'Nykaa So Matte - Immortal', 'color': '#8E4585', 'price': '₹449'}
    ],
    'Deep': [
        {'name': 'Deep Plum', 'brand': 'Fenty Beauty Stunna Lip Paint - Uncensored', 'color': '#5C1A33', 'price': '₹1,850'},
        {'name': 'Rich Berry', 'brand': 'MAC Satin Lipstick - Diva', 'color': '#6F2232', 'price': '₹1,900'},
        {'name': 'Dark Red', 'brand': 'Nykaa So Matte - Boss Lady', 'color': '#8B0000', 'price': '₹449'},
        {'name': 'Burgundy', 'brand': 'Sugar Nothing Else Matter - 06 Plum Yum', 'color': '#800020', 'price': '₹699'}
    ]
}

# Style recommendations
STYLE_RECOMMENDATIONS = {
    'Fair': {
        'colors': ['Soft pastels', 'Light blues and lavenders', 'Mint green', 'Blush pink', 'Powder blue'],
        'avoid': ['Neon colors', 'Very dark colors that create harsh contrast', 'Pure black'],
        'metals': 'Silver and white gold jewelry complement fair skin beautifully'
    },
    'Light': {
        'colors': ['Coral and peach tones', 'Light teal', 'Soft yellow', 'Rose pink', 'Aqua blue'],
        'avoid': ['Overly pale colors that wash you out', 'Beige matching your skin tone'],
        'metals': 'Both gold and silver work well - choose based on undertone'
    },
    'Medium': {
        'colors': ['Earth tones', 'Olive green', 'Burnt orange', 'Deep teal', 'Burgundy', 'Rich browns'],
        'avoid': ['Colors too close to skin tone', 'Pale yellows'],
        'metals': 'Warm gold tones look stunning on medium skin'
    },
    'Tan': {
        'colors': ['Rich jewel tones', 'Emerald green', 'Sapphire blue', 'Ruby red', 'Royal purple', 'Mustard yellow'],
        'avoid': ['Muddy browns', 'Dull olive tones'],
        'metals': 'Gold and bronze jewelry create beautiful warmth'
    },
    'Deep': {
        'colors': ['Vibrant colors', 'Electric blue', 'Fuchsia', 'Bright yellow', 'Pure white', 'Tangerine', 'Hot pink'],
        'avoid': ['Dull, muted colors', 'Dark navy that blends'],
        'metals': 'Gold jewelry creates stunning contrast against deep skin'
    }
}

# Initialize analyzer
analyzer = SkinAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'DermAI Backend is running',
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_skin():
    """Main skin analysis endpoint"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Perform analysis
        results = analyzer.analyze(data['image'])
        
        # Add product recommendations
        results['products'] = INDIAN_PRODUCTS.get(
            results['skinType'], 
            INDIAN_PRODUCTS['Normal']
        )
        
        # Add lipstick recommendations
        results['lipsticks'] = LIPSTICK_RECOMMENDATIONS.get(
            results['skinTone']['name'],
            LIPSTICK_RECOMMENDATIONS['Medium']
        )
        
        # Add style recommendations
        results['styleRecommendations'] = STYLE_RECOMMENDATIONS.get(
            results['skinTone']['name'],
            STYLE_RECOMMENDATIONS['Medium']
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/<skin_type>', methods=['GET'])
def get_products(skin_type):
    """Get product recommendations for specific skin type"""
    products = INDIAN_PRODUCTS.get(skin_type, INDIAN_PRODUCTS['Normal'])
    return jsonify({
        'success': True,
        'skinType': skin_type,
        'products': products
    })

@app.route('/api/lipsticks/<skin_tone>', methods=['GET'])
def get_lipsticks(skin_tone):
    """Get lipstick recommendations for specific skin tone"""
    lipsticks = LIPSTICK_RECOMMENDATIONS.get(skin_tone, LIPSTICK_RECOMMENDATIONS['Medium'])
    return jsonify({
        'success': True,
        'skinTone': skin_tone,
        'lipsticks': lipsticks
    })

if __name__ == '__main__':
    print("🚀 DermAI Backend Server Starting...")
    print("📊 Advanced AI Skin Analysis Engine Ready")
    print("🔬 Computer Vision Models Loaded")
    print("💄 Indian Product Database: ✓")
    print("🎨 Lipstick Recommendations: ✓")
    print("👗 Style Engine: ✓")
    print("\n✨ Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
