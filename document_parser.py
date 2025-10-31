"""
Document Parser - Extracts requirements from complex documents
Can work with or without LLM
"""

import re
import json

class RequirementsDocumentParser:
    """Parse complex documents into structured requirements"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.use_llm = True
            print("✅ LLM parser enabled (Claude)")
        else:
            self.use_llm = False
            print("⚠️  Using fallback rule-based parser (no API key)")
    
    def parse_document(self, document_text):
        """
        Parse document and extract key requirements
        
        Returns:
            Dict with extracted features
        """
        if self.use_llm:
            return self._parse_with_llm(document_text)
        else:
            return self._parse_with_rules(document_text)
    
    def _parse_with_llm(self, doc_text):
        """Use Claude to parse document"""
        
        prompt = f"""Analyze this technical requirements document and extract key information.

Document:
{doc_text}

Extract and return as JSON:
{{
  "simple_description": "2-3 sentence summary suitable for AWS architecture recommendation",
  "scale_indicators": {{
    "users": <number or null>,
    "storage_tb": <number or null>,
    "concurrency": <number or null>
  }},
  "technical_features": ["list", "of", "features"],
  "compute_preference": "serverless|traditional|gpu|mixed",
  "budget_monthly": <number or null>
}}

Features to look for: video_streaming, ai_ml, real_time, authentication, file_upload, 
database, messaging, event_driven, analytics, mobile_backend, static_hosting

Be precise and concise."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback
            return self._parse_with_rules(doc_text)
    
    def _parse_with_rules(self, doc_text):
        """Fallback: Rule-based parsing"""
        
        doc_lower = doc_text.lower()
        
        # Extract numbers
        users = self._extract_number(doc_text, r'(\d+[,\d]*)\s*(?:daily\s*)?(?:active\s*)?users')
        storage = self._extract_number(doc_text, r'(\d+)\s*tb')
        concurrency = self._extract_number(doc_text, r'(\d+[,\d]*)\s*(?:peak\s*)?concurrency')
        budget = self._extract_number(doc_text, r'\$(\d+[,\d]*)')
        
        # Detect features
        features = []
        feature_keywords = {
            'video_streaming': ['video', 'streaming', 'lecture', 'media'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'recommendation', 'nlp'],
            'real_time': ['real-time', 'realtime', 'live', 'websocket'],
            'authentication': ['auth', 'login', 'user management', 'signup'],
            'file_upload': ['upload', 'file storage', 'document'],
            'database': ['database', 'data storage', 'sql', 'nosql'],
            'messaging': ['messaging', 'notification', 'email', 'sms', 'push'],
            'event_driven': ['event', 'queue', 'asynchronous'],
            'analytics': ['analytics', 'reporting', 'dashboard', 'metrics'],
            'mobile_backend': ['mobile', 'app', 'ios', 'android'],
            'static_hosting': ['static', 'html', 'css', 'website']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(kw in doc_lower for kw in keywords):
                features.append(feature)
        
        # Compute preference
        compute_pref = "serverless"
        if 'gpu' in doc_lower or 'training' in doc_lower:
            compute_pref = "gpu"
        elif 'ec2' in doc_lower or 'server' in doc_lower:
            compute_pref = "traditional"
        elif 'serverless' in doc_lower or 'lambda' in doc_lower:
            compute_pref = "serverless"
        
        # Generate simple description
        simple_desc = self._generate_simple_description(features, users, compute_pref)
        
        return {
            "simple_description": simple_desc,
            "scale_indicators": {
                "users": users,
                "storage_tb": storage,
                "concurrency": concurrency
            },
            "technical_features": features,
            "compute_preference": compute_pref,
            "budget_monthly": budget
        }
    
    def _extract_number(self, text, pattern):
        """Extract number from text using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return int(num_str)
            except:
                return None
        return None
    
    def _generate_simple_description(self, features, users, compute_pref):
        """Generate simple text description from features"""
        
        parts = ["Build a"]
        
        if compute_pref == "serverless":
            parts.append("serverless")
        elif compute_pref == "gpu":
            parts.append("GPU-accelerated")
        
        # Primary purpose
        if "video_streaming" in features:
            parts.append("video streaming platform")
        elif "mobile_backend" in features:
            parts.append("mobile backend")
        elif "static_hosting" in features:
            parts.append("static website")
        else:
            parts.append("web application")
        
        # Add scale
        if users:
            if users < 10000:
                parts.append("for small-scale users")
            elif users < 50000:
                parts.append("for medium-scale users")
            else:
                parts.append("for large-scale users")
        
        # Add key features
        feature_mentions = []
        if "authentication" in features:
            feature_mentions.append("user authentication")
        if "ai_ml" in features:
            feature_mentions.append("AI-powered recommendations")
        if "real_time" in features:
            feature_mentions.append("real-time interactions")
        if "file_upload" in features:
            feature_mentions.append("file storage")
        
        if feature_mentions:
            parts.append("with " + ", ".join(feature_mentions[:3]))
        
        return " ".join(parts) + "."


# Test
if __name__ == "__main__":
    parser = RequirementsDocumentParser(api_key=None)
    
    test_doc = """
    EduVision - Online learning platform
    - 50,000 daily users
    - Video streaming for lectures
    - AI-powered recommendations
    - Budget: $5000/month
    """
    
    result = parser.parse_document(test_doc)
    print(json.dumps(result, indent=2))
    print("\n✅ Parser working!")
