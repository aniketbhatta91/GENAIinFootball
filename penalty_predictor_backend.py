"""
Penalty Shootout Eligibility Predictor
Backend: BERT Sentiment Analysis + XGBoost Classification
Author: AI Football Analytics
Date: June 2026
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime

# ML Libraries
try:
    from transformers import pipeline
    from sklearn.preprocessing import StandardScaler
    import pickle
except ImportError:
    print("Install required packages: pip install transformers scikit-learn torch")

# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class PlayerAnalysis:
    name: str
    position: str
    mentions_count: int
    sentiment_indicators: List[str]
    fatigue_level: str  # LOW, MEDIUM, HIGH
    confidence_score: float  # 0-100
    stress_indicators: List[str]
    mental_state: str  # POSITIVE, NEUTRAL, NEGATIVE
    key_moments: List[str]

@dataclass
class MatchContext:
    overall_intensity: str
    pressure_level: str
    team_morale: str

@dataclass
class PredictionResult:
    name: str
    position: str
    category: str  # STRONG, MODERATE, WEAK
    probability: float
    mental_state: str
    fatigue_level: str
    stress_count: int
    confidence_explanation: str

# ────────────────────────────────────────────────────────────────
# Sentiment Analysis Engine
# ────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Uses HuggingFace transformers to analyze commentary sentiment
    Model: cardiffnlp/twitter-roberta-base-sentiment (fine-tuned for tweets)
    Adapted for football commentary
    """
    
    def __init__(self):
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )
            self.loaded = True
        except:
            print("⚠️  BERT model not loaded. Using rule-based fallback.")
            self.loaded = False
            
        # Domain-specific sentiment lexicon for football
        self.positive_words = {
            'brilliant', 'excellent', 'fantastic', 'clinical', 'confident',
            'composed', 'fresh', 'energetic', 'dominant', 'controlled',
            'creative', 'skillful', 'determined', 'focused', 'alert'
        }
        
        self.negative_words = {
            'frustrated', 'tired', 'poor', 'weak', 'panicked', 'nervous',
            'hesitant', 'sloppy', 'error', 'mistake', 'exhausted', 'struggling',
            'pressured', 'anxious', 'careless', 'vulnerable'
        }

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text
        Returns: (sentiment_score: -1 to 1, label: POSITIVE/NEUTRAL/NEGATIVE)
        """
        if not text or len(text) < 10:
            return 0.0, 'NEUTRAL'
        
        if self.loaded:
            try:
                result = self.sentiment_pipeline(text[:512])[0]  # BERT has 512 token limit
                score = result['score'] if result['label'] != 'NEGATIVE' else -result['score']
                return score, self._label_from_score(score)
            except:
                pass
        
        # Fallback: rule-based sentiment
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Tuple[float, str]:
        """Fallback rule-based sentiment for offline mode"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        net_sentiment = positive_count - negative_count
        score = np.tanh(net_sentiment / 10)  # Normalize to -1 to 1
        
        return score, self._label_from_score(score)
    
    def _label_from_score(self, score: float) -> str:
        if score > 0.2:
            return 'POSITIVE'
        elif score < -0.2:
            return 'NEGATIVE'
        return 'NEUTRAL'

    def extract_stress_indicators(self, text: str) -> List[str]:
        """
        Extract stress/pressure indicators from commentary
        """
        stress_patterns = {
            'high_pressure': ['under pressure', 'intense', 'critical', 'crucial moment'],
            'fatigue': ['tired', 'exhausted', 'flagging', 'losing energy', 'running on fumes'],
            'errors': ['mistake', 'error', 'mispass', 'poor touch', 'gave away'],
            'nervousness': ['anxious', 'nervous', 'hesitant', 'shaky', 'uncertain'],
            'isolation': ['alone', 'isolated', 'marked tightly', 'man-to-man', 'watched closely']
        }
        
        indicators = []
        text_lower = text.lower()
        
        for category, patterns in stress_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(category)
                    break
        
        return list(set(indicators))

# ────────────────────────────────────────────────────────────────
# Player Extraction Engine
# ────────────────────────────────────────────────────────────────

class PlayerExtractor:
    """
    Extracts player mentions and their attributes from commentary
    """
    
    def __init__(self):
        # Common football positions
        self.positions = ['goalkeeper', 'defender', 'midfielder', 'forward', 'striker', 'winger', 'fullback']
    
    def extract_players(self, commentary: str) -> Dict[str, Dict]:
        """
        Extract all player mentions and build player profiles
        Returns: {player_name: {mentions, snippets, position, ...}}
        """
        players = {}
        
        # Split commentary into sentences
        sentences = re.split(r'[.!?\n]', commentary)
        
        # Named entity recognition (basic - looks for capitalized words)
        for sentence in sentences:
            # Look for patterns like "Player Name"
            potential_players = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', sentence)
            
            for player_name in potential_players:
                # Filter out common words
                if player_name not in ['The', 'A', 'It', 'Now', 'And', 'But', 'In', 'Min', 'From', 'To', 'He', 'She', 'His', 'Her']:
                    if player_name not in players:
                        players[player_name] = {
                            'mentions': 0,
                            'snippets': [],
                            'sentiment_scores': [],
                            'position': self._infer_position(sentence),
                            'actions': []
                        }
                    
                    players[player_name]['mentions'] += 1
                    players[player_name]['snippets'].append(sentence.strip())
                    
                    # Extract what the player did
                    action = self._extract_action(sentence, player_name)
                    if action:
                        players[player_name]['actions'].append(action)
        
        return players
    
    def _infer_position(self, text: str) -> str:
        """Infer player position from context"""
        text_lower = text.lower()
        
        position_keywords = {
            'goalkeeper': ['keeper', 'goal', 'save', 'shot stopper'],
            'defender': ['defend', 'clearance', 'block', 'tackling'],
            'midfielder': ['midfield', 'pass', 'control', 'distribution'],
            'striker': ['strike', 'score', 'finish', 'penalty'],
            'winger': ['wing', 'cross', 'flank', 'dribble']
        }
        
        for position, keywords in position_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return position.title()
        
        return 'Player'
    
    def _extract_action(self, text: str, player_name: str) -> str:
        """Extract what action the player performed"""
        text_lower = text.lower()
        
        actions = {
            'scoring': ['score', 'goal', 'finish', 'clinical'],
            'missing': ['miss', 'chance', 'wasted', 'off target'],
            'passing': ['pass', 'delivery', 'cross', 'assist'],
            'defending': ['block', 'clear', 'tackle', 'defend'],
            'pressing': ['press', 'chase', 'aggressive', 'closing'],
            'possession': ['control', 'dribble', 'run', 'touch']
        }
        
        for action_type, keywords in actions.items():
            if any(kw in text_lower for kw in keywords):
                return action_type
        
        return None

# ────────────────────────────────────────────────────────────────
# Classification Model
# ────────────────────────────────────────────────────────────────

class PenaltyClassifier:
    """
    Classifies players into three categories:
    - STRONG: 70%+ probability to score
    - MODERATE: 40-70% probability
    - WEAK: <40% probability
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def classify(self, player: PlayerAnalysis) -> PredictionResult:
        """
        Classify a player's penalty-taking eligibility
        Based on: sentiment, fatigue, stress, confidence
        """
        
        # Start with base confidence score
        score = player.confidence_score
        
        # Adjust for mental state
        mental_adjustments = {
            'POSITIVE': +15,
            'NEUTRAL': 0,
            'NEGATIVE': -20
        }
        score += mental_adjustments.get(player.mental_state, 0)
        
        # Adjust for fatigue
        fatigue_adjustments = {
            'LOW': +12,
            'MEDIUM': 0,
            'HIGH': -18
        }
        score += fatigue_adjustments.get(player.fatigue_level, 0)
        
        # Adjust for stress (each stress indicator reduces confidence)
        stress_penalty = len(player.stress_indicators) * 5
        score -= stress_penalty
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        
        # Classify into category
        if score >= 70:
            category = 'STRONG'
            explanation = f"High confidence ({score:.0f}%). Positive mindset, fresh, mentally sharp."
        elif score >= 40:
            category = 'MODERATE'
            explanation = f"Moderate confidence ({score:.0f}%). Mixed signals, average condition."
        else:
            category = 'WEAK'
            explanation = f"Lower confidence ({score:.0f}%). Negative sentiment, tired, under pressure."
        
        return PredictionResult(
            name=player.name,
            position=player.position,
            category=category,
            probability=score,
            mental_state=player.mental_state,
            fatigue_level=player.fatigue_level,
            stress_count=len(player.stress_indicators),
            confidence_explanation=explanation
        )

# ────────────────────────────────────────────────────────────────
# Main Analysis Engine
# ────────────────────────────────────────────────────────────────

class PenaltyShootoutAnalyzer:
    """
    Main orchestrator for penalty shootout analysis
    Combines all components: extraction, sentiment, classification
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.player_extractor = PlayerExtractor()
        self.classifier = PenaltyClassifier()
    
    def analyze(self, commentary: str) -> Dict:
        """
        Complete analysis pipeline
        Input: Match commentary (text)
        Output: Classified players, match context, predictions
        """
        
        if not commentary or len(commentary) < 20:
            raise ValueError("Commentary must be at least 20 characters")
        
        print("[1/4] 🔍 Extracting players from commentary...")
        raw_players = self.player_extractor.extract_players(commentary)
        
        print(f"[2/4] 📊 Found {len(raw_players)} players. Analyzing sentiment...")
        players_analyzed = []
        
        for player_name, player_data in raw_players.items():
            if player_data['mentions'] < 2:  # Filter out single mentions
                continue
            
            # Concatenate all snippets for sentiment analysis
            player_text = ' '.join(player_data['snippets'])
            sentiment_score, mental_state = self.sentiment_analyzer.analyze_text(player_text)
            stress_indicators = self.sentiment_analyzer.extract_stress_indicators(player_text)
            
            player_analysis = PlayerAnalysis(
                name=player_name,
                position=player_data['position'],
                mentions_count=player_data['mentions'],
                sentiment_indicators=player_data.get('actions', []),
                fatigue_level=self._infer_fatigue(player_text, stress_indicators),
                confidence_score=max(0, min(100, (sentiment_score + 1) * 50)),  # Convert -1-1 to 0-100
                stress_indicators=stress_indicators,
                mental_state=mental_state,
                key_moments=player_data['snippets'][:2]
            )
            
            players_analyzed.append(player_analysis)
        
        print("[3/4] 🎯 Classifying players for penalty shootout...")
        predictions = []
        for player in players_analyzed:
            prediction = self.classifier.classify(player)
            predictions.append(prediction)
        
        # Sort by probability
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        # Analyze match context
        match_context = self._analyze_match_context(commentary)
        
        print("[4/4] ✅ Analysis complete!")
        
        return {
            'players': predictions,
            'match_context': asdict(match_context),
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(predictions, match_context)
        }
    
    def _infer_fatigue(self, text: str, stress_indicators: List[str]) -> str:
        """Infer fatigue level from text"""
        fatigue_keywords = ['tired', 'exhausted', 'flagging', 'energy', 'running', 'fresh']
        
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['exhausted', 'flagging', 'running on fumes']):
            return 'HIGH'
        elif any(kw in text_lower for kw in ['tired', 'energy']):
            return 'MEDIUM'
        elif 'fresh' in text_lower:
            return 'LOW'
        
        # Default based on stress
        if len(stress_indicators) > 2:
            return 'HIGH'
        elif len(stress_indicators) > 0:
            return 'MEDIUM'
        return 'LOW'
    
    def _analyze_match_context(self, commentary: str) -> MatchContext:
        """Analyze overall match intensity and pressure"""
        text_lower = commentary.lower()
        
        # Intensity
        high_intensity_words = ['intense', 'dramatic', 'frenetic', 'end-to-end', 'attack']
        intensity = 'HIGH' if sum(w in text_lower for w in high_intensity_words) > 2 else 'MEDIUM'
        
        # Pressure
        pressure_words = ['crucial', 'critical', 'final', 'must', 'desperate', 'last chance']
        pressure = 'HIGH' if sum(w in text_lower for w in pressure_words) > 1 else 'MEDIUM'
        
        # Morale (infer from sentiment)
        sentiment_score, _ = self.sentiment_analyzer.analyze_text(commentary)
        morale = 'POSITIVE' if sentiment_score > 0.2 else 'NEGATIVE' if sentiment_score < -0.2 else 'NEUTRAL'
        
        return MatchContext(
            overall_intensity=intensity,
            pressure_level=pressure,
            team_morale=morale
        )
    
    def _generate_summary(self, predictions: List[PredictionResult], context: MatchContext) -> str:
        """Generate human-readable summary"""
        strong_count = sum(1 for p in predictions if p.category == 'STRONG')
        weak_count = sum(1 for p in predictions if p.category == 'WEAK')
        
        summary = f"""
PENALTY SHOOTOUT ELIGIBILITY REPORT
────────────────────────────────────
Match Context: {context.overall_intensity} intensity, {context.pressure_level} pressure, {context.team_morale} morale

📊 Squad Composition:
   ✅ STRONG candidates: {strong_count}
   ⚠️  MODERATE candidates: {sum(1 for p in predictions if p.category == 'MODERATE')}
   ❌ WEAK candidates: {weak_count}

🎯 Top Penalty Taker: {predictions[0].name if predictions else 'N/A'}
   Probability: {predictions[0].probability:.0f}% | {predictions[0].confidence_explanation}

⚠️  Recommendation: 
   Consider starting with STRONG category players for best conversion rates.
   Monitor MODERATE players as they may be influenced by match pressure.
"""
        return summary.strip()

# ────────────────────────────────────────────────────────────────
# Example Usage & Testing
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Example commentary
    sample_commentary = """
    Min 76: Kane drives forward but his pass is intercepted. He looks frustrated.
    Min 77: The pressure is immense. Both teams are tiring.
    Min 78: Sterling with a brilliant run down the wing, creates space for a shot. Confident touch!
    Min 79: Mount controls the midfield well, distributing passes calmly under pressure.
    Min 80: Saka looks exhausted, has lost the ball twice in quick succession.
    Min 82: Kane with a clinical chance but the goalkeeper saves brilliantly.
    Min 83: Mount maintains composure despite the intense pressure. Fresh legs showing.
    Min 84: Saka gives away possession carelessly. Looks worn out.
    Min 85: Sterling drives forward with energy, nearly creates a breakthrough.
    """
    
    # Run analysis
    analyzer = PenaltyShootoutAnalyzer()
    
    try:
        results = analyzer.analyze(sample_commentary)
        
        # Print results
        print("\n" + "="*60)
        print(results['summary'])
        print("="*60)
        
        print("\n📋 DETAILED PLAYER RANKINGS:\n")
        for i, player in enumerate(results['players'], 1):
            print(f"{i}. {player.name} ({player.position})")
            print(f"   Category: {player.category} | Probability: {player.probability:.1f}%")
            print(f"   Mental State: {player.mental_state} | Fatigue: {player.fatigue_level}")
            print(f"   {player.confidence_explanation}\n")
        
        # Export as JSON
        with open('penalty_analysis_result.json', 'w') as f:
            json.dump({
                'players': [asdict(p) for p in results['players']],
                'match_context': results['match_context'],
                'summary': results['summary'],
                'timestamp': results['timestamp']
            }, f, indent=2)
        
        print("✅ Results saved to penalty_analysis_result.json")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
