"""
Product Category VALIDATOR using Gemini 2.0 Flash
- Validates if products truly belong to their matched category
- Binary classification: YES (belongs) or NO (doesn't belong)
- Parallel processing with rate limiting
- Cost tracking and checkpoint/resume support
- Optimized for 500K+ products
"""

import os
import re
import json
import time
import threading
import traceback
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ============================ COST TRACKING ============================

@dataclass
class CostTracker:
    """Tracks API costs and token usage in real-time"""
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    input_cost_per_1m: float = 0.10  # $0.10 per 1M input tokens
    output_cost_per_1m: float = 0.40  # $0.40 per 1M output tokens
    lock: threading.Lock = None

    def __post_init__(self):
        self.lock = threading.Lock()

    def estimate_input_tokens(self, text: str) -> int:
        """Estimate input tokens (conservative for Persian text)"""
        return len(text) // 3

    def estimate_output_tokens(self, text: str) -> int:
        """Estimate output tokens"""
        return max(20, len(text) // 4)  # Even smaller for YES/NO

    def add_call(self, input_tokens: int, output_tokens: int, success: bool = True) -> Dict[str, float]:
        """Add a call's token usage and return current costs"""
        with self.lock:
            self.input_tokens_total += input_tokens
            self.output_tokens_total += output_tokens
            self.total_calls += 1
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1

            input_cost = (self.input_tokens_total / 1_000_000) * self.input_cost_per_1m
            output_cost = (self.output_tokens_total / 1_000_000) * self.output_cost_per_1m
            total_cost = input_cost + output_cost

            return {
                "input_tokens": self.input_tokens_total,
                "output_tokens": self.output_tokens_total,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "cost_this_call": (input_tokens / 1_000_000 * self.input_cost_per_1m +
                                 output_tokens / 1_000_000 * self.output_cost_per_1m)
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get current cost summary"""
        with self.lock:
            input_cost = (self.input_tokens_total / 1_000_000) * self.input_cost_per_1m
            output_cost = (self.output_tokens_total / 1_000_000) * self.output_cost_per_1m

            return {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "input_tokens": self.input_tokens_total,
                "output_tokens": self.output_tokens_total,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": input_cost + output_cost,
                "avg_cost_per_call": (input_cost + output_cost) / max(1, self.total_calls),
                "success_rate": self.successful_calls / max(1, self.total_calls) * 100
            }


# Global cost tracker instance
cost_tracker = CostTracker()

# ============================ RATE LIMITER ============================

class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call = time.time()


# ============================ CONFIG ============================

# GROUPS TO PROCESS - Edit this list to control which groups to validate and in what order
# Format: (input_filename, output_filename)
# Comment out (add # at start) or remove groups you don't want to process
GROUPS_TO_PROCESS = [
    # ("filtered_group1.csv", "validated_group1.csv"),
    # ("filtered_group2.csv", "validated_group2.csv"),
    # ("filtered_group3.csv", "validated_group3.csv"),
    # ("filtered_group4.csv", "validated_group4.csv"),
    # ("filtered_group5.csv", "validated_group5.csv"),
    # ("filtered_group6.csv", "validated_group6.csv"),
    # ("filtered_group7.csv", "validated_group7.csv"),
    # ("filtered_group8.csv", "validated_group8.csv"),
    ("filtered_group9.csv", "validated_group9.csv"),    
    ("filtered_group10.csv", "validated_group10.csv"),  
    ("filtered_group11.csv", "validated_group11.csv"),
    ("filtered_group12.csv", "validated_group12.csv"), 
]

CONFIG = {
    "GEMINI_API_KEY": os.getenv("GENAI_API_KEY", "AIzaSyBY1JI7IW-mNA2ZmgD2Q5fkbtNpF629HX4"),
    "MODEL_NAME": "gemini-2.5-flash",  # Non-lite version for better accuracy (handle region access separately)
    "TEMPERATURE": 0.3,  # Low temperature for consistency

    # Processing
    "MAX_WORKERS": 5,  # Parallel workers (must be ≤ rate limit to avoid GOAWAY errors)
    "RATE_LIMIT_PER_SEC": 5,  # API calls per second (reduced to avoid quota errors)
    "BATCH_SIZE": 30,  # Number of products to validate in a single API call
    "CHECKPOINT_INTERVAL": 1000,  # Save progress every N products
    "MAX_RETRIES": 2,
    "RETRY_DELAY": 2.0,

    # Cost display
    "SHOW_COST_EVERY_N": 100,

    # Success rate monitoring
    "SUCCESS_RATE_DROP_THRESHOLD": 1.0,  # Pause if success rate drops by this % (1%)
    "PAUSE_DURATION_HOURS": 0.1,  # How long to pause when success rate drops

    # Telegram Bot Configuration
    "TELEGRAM_BOT_TOKEN": "8205938582:AAG-fhOjW4tMPkNRpYU8J_Xg7vgMLisHCBU",  # Get from @BotFather
    "TELEGRAM_CHAT_ID": "-5091693030",  # Your group chat ID
    "TELEGRAM_STATUS_INTERVAL": 60,  # Send status update every 60 seconds
}

# Configure Gemini
genai.configure(api_key=CONFIG["GEMINI_API_KEY"])


# ============================ TELEGRAM NOTIFICATIONS ============================

class TelegramNotifier:
    """Professional Telegram bot for monitoring validation progress"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bot_token != "YOUR_BOT_TOKEN_HERE"

    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send a message to Telegram"""
        if not self.enabled:
            print(f"[TELEGRAM DISABLED] {message}")
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"[TELEGRAM ERROR] Failed to send message: {e}")
            return False

    def send_start_notification(self, group_name: str, total_products: int,
                               estimated_time_min: float, estimated_cost: float):
        """Send start notification"""
        message = f"""
🚀 <b>VALIDATION STARTED</b>

📁 Group: <code>{group_name}</code>
📊 Products: <b>{total_products:,}</b>
⏱ Estimated Time: <b>{estimated_time_min:.1f} min</b>
💰 Estimated Cost: <b>${estimated_cost:.3f}</b>

⚙️ Config:
  • Workers: {CONFIG['MAX_WORKERS']}
  • Batch Size: {CONFIG['BATCH_SIZE']}
  • Rate: {CONFIG['RATE_LIMIT_PER_SEC']} calls/sec

⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_status_update(self, group_name: str, completed: int, total: int,
                          success_rate: float, current_cost: float, elapsed_min: float):
        """Send periodic status update"""
        progress_pct = (completed / total * 100) if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * completed / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)

        message = f"""
📈 <b>STATUS UPDATE</b>

📁 Group: <code>{group_name}</code>
{bar} {progress_pct:.1f}%

✅ Completed: <b>{completed:,}</b> / {total:,}
🎯 Success Rate: <b>{success_rate:.1f}%</b>
💰 Cost So Far: <b>${current_cost:.3f}</b>
⏱ Elapsed: <b>{elapsed_min:.1f} min</b>

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_error_alert(self, group_name: str, error_type: str, error_message: str,
                        completed: int, total: int):
        """Send critical error alert"""
        message = f"""
🚨 <b>CRITICAL ERROR - STOPPED</b>

📁 Group: <code>{group_name}</code>
❌ Error Type: <b>{error_type}</b>

📝 Message:
<code>{error_message[:500]}</code>

Progress when stopped:
  • Completed: {completed:,} / {total:,}
  • {(completed/total*100) if total > 0 else 0:.1f}% done

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <b>Processing stopped to prevent accuracy loss!</b>
"""
        return self.send_message(message.strip())

    def send_completion_notification(self, group_name: str, total: int,
                                    success_count: int, error_count: int,
                                    total_cost: float, elapsed_min: float,
                                    output_file: str):
        """Send completion notification"""
        success_rate = (success_count / total * 100) if total > 0 else 0

        message = f"""
✅ <b>GROUP COMPLETED</b>

📁 Group: <code>{group_name}</code>
📊 Total Products: <b>{total:,}</b>

Results:
  ✅ Success: <b>{success_count:,}</b> ({success_rate:.1f}%)
  ❌ Errors: <b>{error_count:,}</b>

💰 Total Cost: <b>${total_cost:.3f}</b>
⏱ Duration: <b>{elapsed_min:.1f} min</b>

💾 Output: <code>{output_file}</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())


# Global instances
telegram_notifier = TelegramNotifier(
    CONFIG["TELEGRAM_BOT_TOKEN"],
    CONFIG["TELEGRAM_CHAT_ID"]
)

# Global rate limiter
rate_limiter = RateLimiter(CONFIG["RATE_LIMIT_PER_SEC"])

# ============================ EXCEPTIONS ============================

class CriticalAPIError(Exception):
    """Exception raised for critical API errors that should stop processing"""
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type}: {message}")


# ============================ CATEGORY DEFINITIONS ============================

def load_category_definitions(json_path: str = None) -> Dict[str, Dict[str, str]]:
    """Load category definitions from JSON file"""
    if json_path is None:
        # Default path: same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "category_definitions.json")

    if not os.path.exists(json_path):
        print(f"[!] Warning: {json_path} not found. Using empty definitions.")
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            definitions = json.load(f)
        print(f"[OK] Loaded {len(definitions)} category definitions from {os.path.basename(json_path)}")
        return definitions
    except Exception as e:
        print(f"[X] Error loading category definitions: {e}")
        return {}

# System instruction for validation
SYSTEM_INSTRUCTION = """You are a strict product category validator.
Your job is to determine if a product truly belongs to a specific category.
Answer with YES only if the product clearly fits the category definition.
Answer with NO if the product doesn't fit, is borderline, or belongs to a different category.
Be strict and accurate. Always respond with valid JSON."""

# ============================ NORMALIZATION ============================

def normalize_persian(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("ك", "ک").replace("ي", "ی")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================ LLM VALIDATOR ============================

class ProductCategoryValidator:
    def __init__(self, category_definitions: Dict[str, Dict[str, str]]):
        self.category_definitions = category_definitions
        # Don't use system_instruction or response_mime_type for compatibility with older SDK
        self.model = genai.GenerativeModel(
            CONFIG["MODEL_NAME"],
            generation_config={
                "temperature": CONFIG["TEMPERATURE"]
            }
        )

    def validate(self, product: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Validate if a product belongs to its assigned category.

        Args:
            product: Dict with product fields
            category: Category to validate against

        Returns:
            Dict with validation result (YES/NO), confidence, and reason
        """
        # Get category definition
        category_info = self.category_definitions.get(category, {})
        if not category_info:
            return {
                "validation": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Category not found",
                "api_time_ms": 0
            }

        # Build product text - use only product_full_name as requested
        product_full_name = normalize_persian(str(product.get('product_full_name', '')))
        product_desc = normalize_persian(str(product.get('product_description', '')))
        business_line = str(product.get('business_line', ''))

        # Combine product text
        product_text = product_full_name
        if product_desc:
            product_text += f" ({product_desc})"

        # Get explanation and exclusions
        explanation = category_info.get('explanation', '')
        exclusions = category_info.get('exclusions', '')

        # Build detailed validation prompt with system instruction included
        prompt = f"""You are a strict product category validator. Your job is to determine if a product truly belongs to a specific category. Answer with YES only if the product clearly fits the category definition. Answer with NO if the product doesn't fit, is borderline, or belongs to a different category. Be strict and accurate.

Product: {product_text}
Business Line: {business_line}

Category: {category}

[OK] INCLUDES: {explanation}

[X] EXCLUDES: {exclusions}

Does this product belong to "{category}" category based on the rules above?
Answer YES only if it clearly matches the INCLUDES rules and does NOT match any EXCLUDES rules.

Return only valid JSON:
{{
  "validation": "YES or NO",
  "confidence": 0.0-1.0,
  "reason": "one sentence explaining why"
}}"""

        # Estimate input tokens
        estimated_input_tokens = cost_tracker.estimate_input_tokens(prompt)

        # Rate limiting
        rate_limiter.wait()

        # Call LLM with retry logic
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                t0 = time.time()
                response = self.model.generate_content(prompt)
                txt = getattr(response, "text", "")

                # Parse JSON response (strict)
                data = self._parse_json(txt)

                # Estimate output tokens
                estimated_output_tokens = cost_tracker.estimate_output_tokens(txt)

                # Track cost
                cost_info = cost_tracker.add_call(estimated_input_tokens, estimated_output_tokens, success=True)

                # Validate response
                if not isinstance(data, dict) or "validation" not in data:
                    raise ValueError("Invalid response format")

                # Normalize validation field
                validation = str(data.get("validation", "NO")).upper()
                if validation not in ["YES", "NO"]:
                    validation = "NO"

                # Add metadata
                data["validation"] = validation
                data.setdefault("confidence", 0.7)
                data.setdefault("reason", "")
                data["api_time_ms"] = int((time.time() - t0) * 1000)

                return data

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                # Categorize error type
                if "goaway" in error_msg.lower() or "enhance_your_calm" in error_msg.lower() or "client_misbehavior" in error_msg.lower():
                    error_category = "GOAWAY_RATE_LIMIT"
                elif "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    error_category = "RATE_LIMIT"
                elif "403" in error_msg or "permission" in error_msg.lower() or "banned" in error_msg.lower():
                    error_category = "PERMISSION"
                elif "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    error_category = "CONNECTION"
                elif "401" in error_msg or "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    error_category = "AUTH"
                else:
                    error_category = "OTHER"

                # Log detailed error
                print(f"\n[ERROR] Attempt {attempt + 1}/{CONFIG['MAX_RETRIES']}")
                print(f"   Type: {error_type}")
                print(f"   Category: {error_category}")
                print(f"   Message: {error_msg[:200]}")

                if attempt == CONFIG["MAX_RETRIES"] - 1:
                    # Final failure - log full details
                    print(f"\n[!] FINAL FAILURE after {CONFIG['MAX_RETRIES']} attempts")
                    print(f"   Error Category: {error_category}")
                    print(f"   Full Error: {error_msg[:500]}")

                    cost_tracker.add_call(estimated_input_tokens, 10, success=False)
                    return {
                        "validation": "ERROR",
                        "confidence": 0.0,
                        "reason": f"{error_category}: {error_msg[:50]}",
                        "api_time_ms": 0,
                        "error_type": error_type,
                        "error_category": error_category
                    }

                # Exponential backoff for GOAWAY errors
                if error_category == "GOAWAY_RATE_LIMIT":
                    backoff_time = CONFIG["RETRY_DELAY"] * (2 ** attempt)  # 0.6, 1.2, 2.4 seconds
                    print(f"   [BACKOFF] Waiting {backoff_time:.1f}s before retry...")
                    time.sleep(backoff_time)
                else:
                    time.sleep(CONFIG["RETRY_DELAY"])

        return {
            "validation": "ERROR",
            "confidence": 0.0,
            "reason": "Max retries exceeded",
            "api_time_ms": 0
        }

    def _parse_json(self, text: str) -> Any:
        """
        Parse JSON from response text.

        STRICT VERSION:
        - Try direct JSON
        - Try code blocks
        - Try substring between first '{' and last '}'
        - If all fail, raise ValueError so caller can treat as error & retry
        """
        text = (text or "").strip()

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strategy 2: Extract from markdown code blocks
        try:
            lower = text.lower()
            if "```json" in lower:
                start = lower.find("```json") + 7
                end = text.find("```", start)
                if end != -1:
                    candidate = text[start:end].strip()
                    return json.loads(candidate)
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                if end != -1:
                    candidate = text[start:end].strip()
                    return json.loads(candidate)
        except Exception:
            pass

        # Strategy 3: Find JSON object/array boundaries
        try:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            first_bracket = text.find("[")
            last_bracket = text.rfind("]")

            candidates = []
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                candidates.append(text[first_brace:last_brace+1])
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                candidates.append(text[first_bracket:last_bracket+1])

            for cand in candidates:
                try:
                    return json.loads(cand)
                except Exception:
                    continue
        except Exception:
            pass

        # If we reach here, we couldn't parse JSON reliably
        raise ValueError("Failed to parse JSON from model response")

    def validate_batch(self, products_with_categories: List[tuple]) -> List[Dict[str, Any]]:
        """
        Validate multiple products in a single API call (batch processing).

        Args:
            products_with_categories: List of (product_dict, category) tuples

        Returns:
            List of validation results, one per product (aligned with products_with_categories)
        """
        if not products_with_categories:
            return []

        # Build structured info aligned with batch order
        products_info = []
        for batch_index, (product, category) in enumerate(products_with_categories):
            category_info = self.category_definitions.get(category, {})
            if not category_info:
                # Category definition missing → mark as UNKNOWN for this item
                products_info.append({
                    "type": "unknown",
                    "batch_index": batch_index,
                    "result": {
                        "validation": "UNKNOWN",
                        "confidence": 0.0,
                        "reason": "Category not found",
                        "api_time_ms": 0
                    }
                })
                continue

            product_full_name = normalize_persian(str(product.get('product_full_name', '')))
            product_desc = normalize_persian(str(product.get('product_description', '')))
            business_line = str(product.get('business_line', ''))

            product_text = product_full_name
            if product_desc:
                product_text += f" ({product_desc})"

            explanation = category_info.get('explanation', '')
            exclusions = category_info.get('exclusions', '')

            products_info.append({
                "type": "valid",
                "batch_index": batch_index,
                "product": product_text,
                "business_line": business_line,
                "category": category,
                "explanation": explanation,
                "exclusions": exclusions
            })

        # Extract only valid products for API call
        valid_entries = [p for p in products_info if p["type"] == "valid"]

        if not valid_entries:
            # All products were UNKNOWN (no category definition)
            results = [None] * len(products_with_categories)
            for p in products_info:
                if p["type"] == "unknown":
                    results[p["batch_index"]] = p["result"]
            return results

        # Build batch prompt with sequential product numbers 1..N
        prompt = """You are a strict product category validator. Validate each product below and return a JSON array with results.

For each product, answer YES only if it clearly matches the INCLUDES rules and does NOT match any EXCLUDES rules. Be strict and accurate.

"""

        for seq_id, p in enumerate(valid_entries, start=1):
            prompt += f"""
Product #{seq_id}:
- Name: {p['product']}
- Business Line: {p['business_line']}
- Category: {p['category']}
- INCLUDES: {p['explanation']}
- EXCLUDES: {p['exclusions']}

"""

        prompt += """
Return a JSON array with one object per product, in the exact same order:
[
  {
    "validation": "YES or NO",
    "confidence": 0.0-1.0,
    "reason": "one sentence explaining why"
  },
  ...
]"""

        # Estimate input tokens
        estimated_input_tokens = cost_tracker.estimate_input_tokens(prompt)

        # Rate limiting
        rate_limiter.wait()

        # Call LLM with retry logic
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                t0 = time.time()
                response = self.model.generate_content(prompt)
                txt = getattr(response, "text", "")

                # Parse JSON array response (strict)
                data = self._parse_json(txt)

                # Estimate output tokens
                estimated_output_tokens = cost_tracker.estimate_output_tokens(txt)

                # Track cost
                cost_tracker.add_call(estimated_input_tokens, estimated_output_tokens, success=True)

                api_time = int((time.time() - t0) * 1000)

                # Normalize response to list
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    raise ValueError("Batch response is not a JSON array")

                # Build results for valid entries by order
                valid_results = []
                for i, p in enumerate(valid_entries):
                    if i < len(data):
                        r = data[i]
                    else:
                        r = {}

                    validation = str(r.get("validation", "NO")).upper()
                    if validation not in ["YES", "NO"]:
                        validation = "NO"

                    valid_results.append({
                        "validation": validation,
                        "confidence": r.get("confidence", 0.7),
                        "reason": r.get("reason", ""),
                        "api_time_ms": api_time
                    })

                # Merge UNKNOWN + valid results back to full batch order
                results = [None] * len(products_with_categories)

                # Fill UNKNOWNs
                for p in products_info:
                    if p["type"] == "unknown":
                        results[p["batch_index"]] = p["result"]

                # Fill valids (align by order of valid_entries)
                for i, p in enumerate(valid_entries):
                    batch_index = p["batch_index"]
                    results[batch_index] = valid_results[i]

                return results

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                # Categorize error type
                if "goaway" in error_msg.lower() or "enhance_your_calm" in error_msg.lower() or "client_misbehavior" in error_msg.lower():
                    error_category = "GOAWAY_RATE_LIMIT"
                elif "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    error_category = "RATE_LIMIT"
                elif "403" in error_msg or "permission" in error_msg.lower() or "banned" in error_msg.lower():
                    error_category = "PERMISSION"
                elif "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    error_category = "CONNECTION"
                elif "401" in error_msg or "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    error_category = "AUTH"
                else:
                    error_category = "OTHER"

                print(f"\n[ERROR] Batch validation attempt {attempt + 1}/{CONFIG['MAX_RETRIES']}")
                print(f"   Type: {error_type}")
                print(f"   Category: {error_category}")
                print(f"   Message: {error_msg[:200]}")

                # CRITICAL: Stop immediately on 403 or 429 errors to prevent accuracy loss
                if error_category in ["PERMISSION", "RATE_LIMIT"]:
                    print(f"\n[!] CRITICAL ERROR DETECTED: {error_category}")
                    print(f"[!] STOPPING PROCESSING TO PREVENT ACCURACY LOSS")
                    raise CriticalAPIError(error_category, error_msg)

                if attempt == CONFIG["MAX_RETRIES"] - 1:
                    print(f"\n[!] FINAL FAILURE after {CONFIG['MAX_RETRIES']} attempts")
                    print(f"   Error Category: {error_category}")
                    cost_tracker.add_call(estimated_input_tokens, 10, success=False)

                    # Return error for all products
                    return [{
                        "validation": "ERROR",
                        "confidence": 0.0,
                        "reason": f"{error_category}: {error_msg[:50]}",
                        "api_time_ms": 0,
                        "error_type": error_type,
                        "error_category": error_category
                    } for _ in products_with_categories]

                # Exponential backoff for GOAWAY errors
                if error_category == "GOAWAY_RATE_LIMIT":
                    backoff_time = CONFIG["RETRY_DELAY"] * (2 ** attempt)
                    print(f"   [BACKOFF] Waiting {backoff_time:.1f}s before retry...")
                    time.sleep(backoff_time)
                else:
                    time.sleep(CONFIG["RETRY_DELAY"])

        # Max retries exceeded
        return [{
            "validation": "ERROR",
            "confidence": 0.0,
            "reason": "Max retries exceeded",
            "api_time_ms": 0
        } for _ in products_with_categories]


# ============================ CHECKPOINT SYSTEM ============================

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, file_key: str, df: pd.DataFrame, index: int):
        """Save checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{file_key}_checkpoint.csv")
        df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")

        # Save metadata
        meta_path = os.path.join(self.checkpoint_dir, f"{file_key}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "last_processed_index": index,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cost_summary": cost_tracker.get_summary()
            }, f, indent=2)

    def load(self, file_key: str) -> tuple[Optional[pd.DataFrame], int]:
        """Load checkpoint if exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{file_key}_checkpoint.csv")
        meta_path = os.path.join(self.checkpoint_dir, f"{file_key}_meta.json")

        if not os.path.exists(checkpoint_path):
            return None, 0

        try:
            df = pd.read_csv(checkpoint_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            print(f"  📂 Checkpoint found: {meta['last_processed_index']} products processed (index marker)")
            print(f"     Last saved: {meta['timestamp']}")
            if 'cost_summary' in meta:
                print(f"     Cost so far: ${meta['cost_summary'].get('total_cost_usd', 0):.4f}")
            return df, meta['last_processed_index']
        except Exception as e:
            print(f"  [!] Failed to load checkpoint: {e}")
            return None, 0

    def clear(self, file_key: str):
        """Clear checkpoint after completion"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{file_key}_checkpoint.csv")
        meta_path = os.path.join(self.checkpoint_dir, f"{file_key}_meta.json")

        for path in [checkpoint_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)


# ============================ PARALLEL PROCESSOR ============================

def validate_product_wrapper(args):
    """Wrapper for parallel validation (single product - deprecated)"""
    idx, row, validator, category = args
    result = validator.validate(row.to_dict(), category)
    return idx, result


def validate_batch_wrapper(args):
    """Wrapper for batch validation"""
    batch_tasks, validator = args
    # Extract products and categories from batch
    products_with_categories = [(row.to_dict(), category) for idx, row, _, category in batch_tasks]

    # Get batch results
    batch_results = validator.validate_batch(products_with_categories)

    # Match results back to indices
    results = []
    for i, (idx, row, _, category) in enumerate(batch_tasks):
        results.append((idx, batch_results[i]))

    return results


def process_file_parallel(input_path: str, output_path: str, validator: ProductCategoryValidator,
                         checkpoint_mgr: CheckpointManager):
    """Process a single filtered CSV file with parallel workers"""

    file_key = os.path.basename(input_path).replace('.csv', '')

    # Initialize Telegram notifier
    telegram_notifier = TelegramNotifier(
        bot_token=CONFIG['TELEGRAM_BOT_TOKEN'],
        chat_id=CONFIG['TELEGRAM_CHAT_ID']
    )

    print(f"\n{'='*80}")
    print(f"[FILE] Processing: {os.path.basename(input_path)}")
    print(f"{'='*80}")

    # Load input file
    try:
        df_input = pd.read_csv(input_path)
        print(f"[OK] Loaded {len(df_input):,} products")
    except Exception as e:
        print(f"[X] Error loading file: {e}")
        return

    if df_input.empty:
        print("[!] File is empty. Skipping.")
        return

    # Check for checkpoint
    df_checkpoint, last_index = checkpoint_mgr.load(file_key)

    if df_checkpoint is not None:
        df = df_checkpoint.copy()
        print(f"[OK] Resuming from checkpoint for {file_key}")
    else:
        df = df_input.copy()
        # Initialize result columns if they don't exist
        df['llm_validation'] = None
        df['llm_confidence'] = None
        df['llm_reason'] = None
        df['llm_api_time_ms'] = None

    # Ensure result columns exist (in case of older checkpoints)
    for col in ['llm_validation', 'llm_confidence', 'llm_reason', 'llm_api_time_ms']:
        if col not in df.columns:
            df[col] = None

    # Parse matched categories - use first matched category for validation
    def get_primary_category(cats_str):
        if pd.isna(cats_str):
            return None
        cats = [c.strip() for c in str(cats_str).split(",")]
        return cats[0] if cats else None

    # Prepare tasks for parallel processing
    tasks = []
    processed_mask = df['llm_validation'].notna()

    for idx in range(len(df)):
        if processed_mask.iloc[idx]:
            continue  # already processed in previous run/checkpoint
        row = df.iloc[idx]
        category = get_primary_category(row.get('matched_categories'))
        if category:
            tasks.append((idx, row, validator, category))

    if not tasks:
        print("[!] No products to validate (all already processed or no category).")
        # Still save out as validated file
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] Nothing to do. Saved existing data to {output_path}")
        return

    print(f"\n[AI] Validating with {CONFIG['MAX_WORKERS']} parallel workers...")
    print(f"   Batch size: {CONFIG['BATCH_SIZE']} products per API call")
    print(f"   Rate limit: {CONFIG['RATE_LIMIT_PER_SEC']} calls/sec")
    print(f"   Processing {len(tasks):,} products in {(len(tasks) + CONFIG['BATCH_SIZE'] - 1) // CONFIG['BATCH_SIZE']:,} batches...")

    # Estimate time and cost
    num_batches = (len(tasks) + CONFIG['BATCH_SIZE'] - 1) // CONFIG['BATCH_SIZE']
    estimated_time_sec = num_batches / CONFIG['RATE_LIMIT_PER_SEC']
    estimated_time_min = estimated_time_sec / 60
    estimated_cost = estimate_cost(len(tasks))['estimated_total_cost']

    # Send Telegram start notification
    telegram_notifier.send_start_notification(
        group_name=file_key,
        total_products=len(tasks),
        estimated_time_min=estimated_time_min,
        estimated_cost=estimated_cost
    )

    # Start time tracking
    start_time = time.time()

    # Periodic status update thread
    stop_status_updates = threading.Event()
    last_status_time = [start_time]  # Use list to allow modification in thread

    results_dict = {}

    def periodic_status_updater():
        while not stop_status_updates.is_set():
            time.sleep(CONFIG['TELEGRAM_STATUS_INTERVAL'])
            if not stop_status_updates.is_set():
                elapsed_min = (time.time() - start_time) / 60
                completed = len(results_dict)
                summary = cost_tracker.get_summary()

                telegram_notifier.send_status_update(
                    group_name=file_key,
                    completed=completed,
                    total=len(tasks),
                    success_rate=summary['success_rate'],
                    current_cost=summary['total_cost_usd'],
                    elapsed_min=elapsed_min
                )
                last_status_time[0] = time.time()

    status_thread = threading.Thread(target=periodic_status_updater, daemon=True)
    status_thread.start()

    # Create batches
    batches = []
    for i in range(0, len(tasks), CONFIG['BATCH_SIZE']):
        batch = tasks[i:i + CONFIG['BATCH_SIZE']]
        batches.append((batch, validator))

    # Process batches in parallel
    baseline_success_rate = None  # Will be set after first 100 calls
    processing_stopped = False
    critical_error = None

    try:
        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            futures = [executor.submit(validate_batch_wrapper, batch) for batch in batches]

            with tqdm(total=len(tasks), desc="Validating", unit="product") as pbar:
                for future in as_completed(futures):
                    try:
                        # Get batch results (list of (idx, result) tuples)
                        batch_results = future.result()

                        # Process each result in the batch
                        for idx, result in batch_results:
                            results_dict[idx] = result

                            # Update dataframe
                            df.at[idx, 'llm_validation'] = result.get('validation', 'ERROR')
                            df.at[idx, 'llm_confidence'] = result.get('confidence', 0.0)
                            df.at[idx, 'llm_reason'] = result.get('reason', '')
                            df.at[idx, 'llm_api_time_ms'] = result.get('api_time_ms', 0)

                            # Also store error metadata if present
                            if 'error_type' in result:
                                df.at[idx, 'error_type'] = result.get('error_type', '')
                            if 'error_category' in result:
                                df.at[idx, 'error_category'] = result.get('error_category', '')

                            pbar.update(1)

                        # Show cost periodically
                        if len(results_dict) % CONFIG["SHOW_COST_EVERY_N"] == 0:
                            summary = cost_tracker.get_summary()
                            current_success_rate = summary['success_rate']

                            pbar.set_postfix_str(
                                f"${summary['total_cost_usd']:.3f} | "
                                f"Success: {current_success_rate:.1f}%"
                            )

                            # Set baseline after first check (100 products)
                            if baseline_success_rate is None and len(results_dict) >= 100:
                                baseline_success_rate = current_success_rate
                                print(f"\n[MONITOR] Baseline success rate set: {baseline_success_rate:.1f}%")

                            # Check if success rate dropped by threshold
                            if baseline_success_rate is not None:
                                rate_drop = baseline_success_rate - current_success_rate
                                if rate_drop >= CONFIG["SUCCESS_RATE_DROP_THRESHOLD"]:
                                    pause_hours = CONFIG["PAUSE_DURATION_HOURS"]
                                    print(f"\n[!] SUCCESS RATE DROP DETECTED!")
                                    print(f"    Baseline: {baseline_success_rate:.1f}%")
                                    print(f"    Current: {current_success_rate:.1f}%")
                                    print(f"    Drop: {rate_drop:.1f}%")
                                    print(f"\n[PAUSE] Pausing for {pause_hours} hour(s)...")
                                    print(f"    Will resume at: {time.strftime('%H:%M:%S', time.localtime(time.time() + pause_hours * 3600))}")

                                    # Save checkpoint before pausing
                                    checkpoint_mgr.save(file_key, df, len(results_dict))

                                    # Pause
                                    time.sleep(pause_hours * 3600)

                                    print(f"\n[RESUME] Continuing validation...")
                                    # Update baseline to current rate after pause
                                    baseline_success_rate = current_success_rate

                        # Checkpoint
                        if len(results_dict) % CONFIG["CHECKPOINT_INTERVAL"] == 0:
                            checkpoint_mgr.save(file_key, df, len(results_dict))

                    except CriticalAPIError as e:
                        # Re-raise so outer handler can stop processing cleanly
                        raise
                    except Exception as e:
                        print(f"\n[X] Error processing future: {e}")

    except CriticalAPIError as e:
        # CRITICAL ERROR (403/429) - Stop immediately to prevent accuracy loss
        processing_stopped = True
        print(f"\n[!] CRITICAL ERROR - STOPPING PROCESSING")
        print(f"[!] Error Type: {e.error_type}")
        print(f"[!] Message: {e.message}")

        # Save checkpoint before stopping
        checkpoint_mgr.save(file_key, df, len(results_dict))
        print(f"[CHECKPOINT] Saved progress: {len(results_dict)}/{len(tasks)} products")

        # Send Telegram error alert
        elapsed_min = (time.time() - start_time) / 60
        telegram_notifier.send_error_alert(
            group_name=file_key,
            error_type=e.error_type,
            error_message=e.message,
            completed=len(results_dict),
            total=len(tasks)
        )

        # Re-raise to stop all processing for this file
        raise

    except Exception as e:
        # OTHER UNEXPECTED ERRORS
        processing_stopped = True
        print(f"\n[X] UNEXPECTED ERROR - STOPPING PROCESSING")
        print(f"[X] Error: {e}")
        traceback.print_exc()

        # Save checkpoint
        checkpoint_mgr.save(file_key, df, len(results_dict))
        print(f"[CHECKPOINT] Saved progress: {len(results_dict)}/{len(tasks)} products")

        # Send Telegram error alert
        elapsed_min = (time.time() - start_time) / 60
        telegram_notifier.send_error_alert(
            group_name=file_key,
            error_type="UNEXPECTED_ERROR",
            error_message=str(e),
            completed=len(results_dict),
            total=len(tasks)
        )

        raise

    else:
        # SUCCESS - No exceptions occurred
        # Final save
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        checkpoint_mgr.clear(file_key)

        print(f"\n[DONE] Completed: {output_path}")
        print(f"   Total products: {len(df):,}")

        # Show validation statistics
        if 'llm_validation' in df.columns:
            print(f"\n[STATS] Validation Results:")
            counts = df['llm_validation'].value_counts()
            for validation, count in counts.items():
                pct = (count / len(df)) * 100
                print(f"   - {validation}: {count:,} ({pct:.1f}%)")

            # Show error breakdown if there are errors
            if 'error_category' in df.columns:
                error_rows = df[df['llm_validation'] == 'ERROR']
                if len(error_rows) > 0:
                    print(f"\n[ERROR BREAKDOWN]")
                    error_cats = error_rows['error_category'].value_counts()
                    for cat, count in error_cats.items():
                        pct = (count / len(error_rows)) * 100
                        print(f"   - {cat}: {count:,} ({pct:.1f}% of errors)")

            print(f"\n[TARGET] Confidence Distribution:")
            if df['llm_confidence'].notna().any():
                high = (df['llm_confidence'] >= 0.8).sum()
                med = ((df['llm_confidence'] >= 0.6) & (df['llm_confidence'] < 0.8)).sum()
                low = (df['llm_confidence'] < 0.6).sum()
                print(f"   - High (≥0.8): {high:,} ({high/len(df)*100:.1f}%)")
                print(f"   - Medium (0.6-0.8): {med:,} ({med/len(df)*100:.1f}%)")
                print(f"   - Low (<0.6): {low:,} ({low/len(df)*100:.1f}%)")

        # Send Telegram completion notification
        elapsed_min = (time.time() - start_time) / 60
        summary = cost_tracker.get_summary()

        success_count = ((df['llm_validation'] == 'YES') | (df['llm_validation'] == 'NO')).sum()
        error_count = (df['llm_validation'] == 'ERROR').sum()

        telegram_notifier.send_completion_notification(
            group_name=file_key,
            total=len(df),
            success_count=success_count,
            error_count=error_count,
            total_cost=summary['total_cost_usd'],
            elapsed_min=elapsed_min,
            output_file=output_path
        )

    finally:
        # ALWAYS stop the status update thread
        stop_status_updates.set()
        print(f"\n[CLEANUP] Stopped status update thread")


# ============================ MAIN ============================

def estimate_cost(total_products: int) -> Dict[str, float]:
    """Estimate total cost for validation"""
    avg_input_tokens = 180  # Smaller prompt for validation
    avg_output_tokens = 20  # YES/NO + short reason

    total_input = total_products * avg_input_tokens
    total_output = total_products * avg_output_tokens

    input_cost = (total_input / 1_000_000) * 0.10
    output_cost = (total_output / 1_000_000) * 0.40

    return {
        "total_products": total_products,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": input_cost + output_cost,
        "cost_per_product": (input_cost + output_cost) / total_products
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "output")
    output_dir = os.path.join(base_dir, "output")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")

    print("=" * 80)
    print("  LLM Product Category VALIDATOR")
    print("=" * 80)
    print(f"Model: {CONFIG['MODEL_NAME']}")
    print(f"Pricing: ${cost_tracker.input_cost_per_1m}/1M input, ${cost_tracker.output_cost_per_1m}/1M output")
    print(f"Workers: {CONFIG['MAX_WORKERS']} parallel")
    print(f"Rate limit: {CONFIG['RATE_LIMIT_PER_SEC']} calls/sec")
    print(f"Checkpoint: Every {CONFIG['CHECKPOINT_INTERVAL']:,} products")
    print("=" * 80)

    # Use the configured groups from GROUPS_TO_PROCESS
    files_to_process = GROUPS_TO_PROCESS
    print(f"\n[CONFIG] Processing {len(files_to_process)} group(s) in order:")
    for i, (input_file, output_file) in enumerate(files_to_process, 1):
        print(f"   {i}. {input_file} -> {output_file}")

    # Count total products
    total_products = 0
    for input_file, _ in files_to_process:
        input_path = os.path.join(input_dir, input_file)
        if os.path.exists(input_path):
            try:
                df = pd.read_csv(input_path)
                total_products += len(df)
            except:
                pass

    if total_products == 0:
        print("\n[!] No filtered files found. Please run category_static_filter.py first.")
        return

    # Estimate cost
    print(f"\n[COST ESTIMATION]")
    print(f"   Total products to validate: {total_products:,}")
    estimate = estimate_cost(total_products)
    print(f"   Estimated tokens: {estimate['estimated_input_tokens']:,} in, {estimate['estimated_output_tokens']:,} out")
    print(f"   Estimated cost: ${estimate['estimated_total_cost']:.2f}")
    print(f"   Cost per product: ${estimate['cost_per_product']:.6f}")

    # Estimate time
    estimated_time_mins = (total_products / CONFIG['RATE_LIMIT_PER_SEC']) / 60
    print(f"   Estimated time: {estimated_time_mins:.1f} minutes ({estimated_time_mins/60:.1f} hours)")

    # Auto-proceed (confirmation disabled for non-interactive mode)
    print("\n" + "="*80)
    print("[START] Auto-proceeding with validation...")

    # Load category definitions from JSON
    print("\n[LOADING] Category definitions...")
    category_definitions = load_category_definitions()
    if not category_definitions:
        print("[X] No category definitions loaded. Please check category_definitions.json")
        return

    # Initialize validator
    print("[LOAD] Initializing LLM validator...")
    validator = ProductCategoryValidator(category_definitions)
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Process files
    for input_file, output_file in files_to_process:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        if not os.path.exists(input_path):
            print(f"\n[!] File not found: {input_file}")
            continue

        try:
            process_file_parallel(input_path, output_path, validator, checkpoint_mgr)
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user. Progress saved to checkpoint.")
            break
        except Exception as e:
            print(f"\n[X] Error processing {input_file}: {e}")
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    summary = cost_tracker.get_summary()
    print(f"[STATS] Total API calls: {summary['total_calls']:,}")
    print(f"   Successful: {summary['successful_calls']:,} ({summary['success_rate']:.1f}%)")
    print(f"   Failed: {summary['failed_calls']:,}")
    print(f"\n[COST] Token Usage:")
    print(f"   Input: {summary['input_tokens']:,} tokens -> ${summary['input_cost_usd']:.4f}")
    print(f"   Output: {summary['output_tokens']:,} tokens -> ${summary['output_cost_usd']:.4f}")
    print(f"\n💵 TOTAL COST: ${summary['total_cost_usd']:.4f}")
    print(f"   Average per call: ${summary['avg_cost_per_call']:.6f}")
    print("=" * 80)

    # Save cost report
    cost_report_path = os.path.join(output_dir, "validation_cost_report.json")
    with open(cost_report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Cost report saved: {cost_report_path}")


if __name__ == "__main__":
    main()
