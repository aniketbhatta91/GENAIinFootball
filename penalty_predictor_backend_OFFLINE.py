"""
Penalty Shootout Eligibility Predictor - SIMPLIFIED VERSION
NO BERT DOWNLOAD - Works Offline Immediately
Uses Rule-Based Sentiment Analysis
"""

import json
import numpy as np
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class PlayerAnalysis:
    name: str
    position: str
    mentions_count: int
    sentiment_indicators: List[str]
    fatigue_level: str
    confidence_score: float
    stress_indicators: List[str]
    mental_state: str
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
    category: str
    probability: float
    mental_state: str
    fatigue_level: str
    stress_count: int
    confidence_explanation: str

# ────────────────────────────────────────────────────────────────
# OFFLINE Sentiment Analyzer (No BERT)
# ────────────────────────────────────────────────────────────────

class OfflineSentimentAnalyzer:
    """
    Rule-based sentiment analysis for football commentary
    NO downloads needed - works immediately offline
    """
    
    def __init__(self):
        print("✅ Offline Sentiment Analyzer Ready (No BERT needed)")
        
        # Sentiment lexicon
        self.positive_words = {
            'brilliant', 'excellent', 'fantastic', 'clinical', 'confident',
            'composed', 'fresh', 'energetic', 'dominant', 'controlled',
            'creative', 'skillful', 'determined', 'focused', 'alert',
            'amazing', 'outstanding', 'great', 'perfect', 'sharp',
            'lively', 'active', 'strong', 'powerful', 'impressive'
        }
        
        self.negative_words = {
            'frustrated', 'tired', 'poor', 'weak', 'panicked', 'nervous',
            'hesitant', 'sloppy', 'error', 'mistake', 'exhausted', 'struggling',
            'pressured', 'anxious', 'careless', 'vulnerable', 'bad',
            'awful', 'terrible', 'horrible', 'slow', 'lazy', 'lost'
        }
        
        self.neutral_words = {
            'play', 'pass', 'move', 'run', 'ball', 'action', 'moment',
            'touch', 'control', 'attempt', 'try'
        }

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text (offline, no BERT)
        Returns: (sentiment_score: -1 to 1, label: POSITIVE/NEUTRAL/NEGATIVE)
        """
        if not text or len(text) < 10:
            return 0.0, 'NEUTRAL'
        
        text_lower = text.lower()
        
        # Count sentiment words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calculate net sentiment
        net_sentiment = positive_count - negative_count
        
        # Normalize to -1 to 1
        if net_sentiment > 0:
            score = min(1.0, net_sentiment / 5.0)
        elif net_sentiment < 0:
            score = max(-1.0, net_sentiment / 5.0)
        else:
            score = 0.0
        
        # Label
        if score > 0.2:
            label = 'POSITIVE'
        elif score < -0.2:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return score, label

    def extract_stress_indicators(self, text: str) -> List[str]:
        """Extract stress/pressure indicators"""
        stress_patterns = {
            'fatigue': ['tired', 'exhausted', 'flagging', 'losing energy', 'fading'],
            'pressure': ['pressure', 'intense', 'critical', 'crucial', 'must score'],
            'errors': ['mistake', 'error', 'mispass', 'poor', 'gave away'],
            'nervousness': ['nervous', 'hesitant', 'uncertain', 'shaky', 'panicked'],
            'isolation': ['alone', 'isolated', 'marked', 'watched']
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
# Player Extraction
# ────────────────────────────────────────────────────────────────

class PlayerExtractor:
    """Extract player mentions from commentary"""
    
    def __init__(self):
        self.positions = ['goalkeeper', 'defender', 'midfielder', 'forward', 'striker', 'winger', 'fullback']
    
    def extract_players(self, commentary: str) -> Dict[str, Dict]:
        """Extract players and their attributes"""
        players = {}
        sentences = re.split(r'[.!?\n]', commentary)
        
        for sentence in sentences:
            # Find capitalized names
            potential_players = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', sentence)
            
            for player_name in potential_players:
                # Filter common words
                if player_name not in ['The', 'A', 'It', 'Now', 'And', 'But', 'In', 'Min', 'From', 'To', 'He', 'She', 'His', 'Her']:
                    if player_name not in players:
                        players[player_name] = {
                            'mentions': 0,
                            'snippets': [],
                            'position': self._infer_position(sentence),
                            'actions': []
                        }
                    
                    players[player_name]['mentions'] += 1
                    players[player_name]['snippets'].append(sentence.strip())
                    
                    action = self._extract_action(sentence, player_name)
                    if action:
                        players[player_name]['actions'].append(action)
        
        return players
    
    def _infer_position(self, text: str) -> str:
        """Infer player position"""
        text_lower = text.lower()
        
        position_keywords = {
            'goalkeeper': ['keeper', 'goal', 'save', 'shot'],
            'defender': ['defend', 'clear', 'block', 'tackle'],
            'midfielder': ['midfield', 'control', 'distribute'],
            'forward': ['strike', 'score', 'finish', 'penalty'],
            'winger': ['wing', 'cross', 'flank', 'dribble']
        }
        
        for position, keywords in position_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return position.title()
        
        return 'Player'
    
    def _extract_action(self, text: str, player_name: str) -> str:
        """Extract action"""
        text_lower = text.lower()
        
        actions = {
            'scoring': ['score', 'goal', 'finish', 'clinical'],
            'missing': ['miss', 'chance', 'wasted'],
            'passing': ['pass', 'cross', 'assist'],
            'defending': ['block', 'clear', 'tackle'],
            'pressing': ['press', 'chase', 'aggressive'],
            'possession': ['control', 'dribble', 'run']
        }
        
        for action_type, keywords in actions.items():
            if any(kw in text_lower for kw in keywords):
                return action_type
        
        return None

# ────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────

class PenaltyClassifier:
    """Classify players into 3 categories"""
    
    def classify(self, player: PlayerAnalysis) -> PredictionResult:
        """Classify player eligibility"""
        
        score = player.confidence_score
        
        # Mental state adjustment
        mental_adjustments = {
            'POSITIVE': +15,
            'NEUTRAL': 0,
            'NEGATIVE': -20
        }
        score += mental_adjustments.get(player.mental_state, 0)
        
        # Fatigue adjustment
        fatigue_adjustments = {
            'LOW': +12,
            'MEDIUM': 0,
            'HIGH': -18
        }
        score += fatigue_adjustments.get(player.fatigue_level, 0)
        
        # Stress penalty
        stress_penalty = len(player.stress_indicators) * 5
        score -= stress_penalty
        
        # Ensure 0-100
        score = max(0, min(100, score))
        
        # Classify
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
# Main Analyzer
# ────────────────────────────────────────────────────────────────

class PenaltyShootoutAnalyzer:
    """Main analyzer - offline, no BERT"""
    
    def __init__(self):
        self.sentiment_analyzer = OfflineSentimentAnalyzer()
        self.player_extractor = PlayerExtractor()
        self.classifier = PenaltyClassifier()
    
    def analyze(self, commentary: str) -> Dict:
        """Complete analysis pipeline"""
        
        if not commentary or len(commentary) < 20:
            raise ValueError("Commentary must be at least 20 characters")
        
        print("[1/4] 🔍 Extracting players...")
        raw_players = self.player_extractor.extract_players(commentary)
        
        print(f"[2/4] 📊 Analyzing {len(raw_players)} players...")
        players_analyzed = []
        
        for player_name, player_data in raw_players.items():
            if player_data['mentions'] < 2:
                continue
            
            player_text = ' '.join(player_data['snippets'])
            sentiment_score, mental_state = self.sentiment_analyzer.analyze_text(player_text)
            stress_indicators = self.sentiment_analyzer.extract_stress_indicators(player_text)
            
            player_analysis = PlayerAnalysis(
                name=player_name,
                position=player_data['position'],
                mentions_count=player_data['mentions'],
                sentiment_indicators=player_data.get('actions', []),
                fatigue_level=self._infer_fatigue(player_text, stress_indicators),
                confidence_score=max(0, min(100, (sentiment_score + 1) * 50)),
                stress_indicators=stress_indicators,
                mental_state=mental_state,
                key_moments=player_data['snippets'][:2]
            )
            
            players_analyzed.append(player_analysis)
        
        print("[3/4] 🎯 Classifying players...")
        predictions = []
        for player in players_analyzed:
            prediction = self.classifier.classify(player)
            predictions.append(prediction)
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        match_context = self._analyze_match_context(commentary)
        
        print("[4/4] ✅ Analysis complete!")
        
        return {
            'players': predictions,
            'match_context': asdict(match_context),
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(predictions, match_context)
        }
    
    def _infer_fatigue(self, text: str, stress_indicators: List[str]) -> str:
        """Infer fatigue level"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['exhausted', 'flagging', 'running on fumes']):
            return 'HIGH'
        elif any(kw in text_lower for kw in ['tired', 'energy']):
            return 'MEDIUM'
        elif 'fresh' in text_lower:
            return 'LOW'
        
        if len(stress_indicators) > 2:
            return 'HIGH'
        elif len(stress_indicators) > 0:
            return 'MEDIUM'
        return 'LOW'
    
    def _analyze_match_context(self, commentary: str) -> MatchContext:
        """Analyze match context"""
        text_lower = commentary.lower()
        
        high_intensity_words = ['intense', 'dramatic', 'frenetic', 'end-to-end', 'attack']
        intensity = 'HIGH' if sum(w in text_lower for w in high_intensity_words) > 2 else 'MEDIUM'
        
        pressure_words = ['crucial', 'critical', 'final', 'must', 'desperate']
        pressure = 'HIGH' if sum(w in text_lower for w in pressure_words) > 1 else 'MEDIUM'
        
        sentiment_score, _ = self.sentiment_analyzer.analyze_text(commentary)
        morale = 'POSITIVE' if sentiment_score > 0.2 else 'NEGATIVE' if sentiment_score < -0.2 else 'NEUTRAL'
        
        return MatchContext(
            overall_intensity=intensity,
            pressure_level=pressure,
            team_morale=morale
        )
    
    def _generate_summary(self, predictions, context):
        """Generate summary"""
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
"""
        return summary.strip()

# ────────────────────────────────────────────────────────────────
# Test
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
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
    
    analyzer = PenaltyShootoutAnalyzer()
    
    try:
        results = analyzer.analyze(sample_commentary)
        
        print("\n" + "="*60)
        print(results['summary'])
        print("="*60)
        
        print("\n📋 DETAILED PLAYER RANKINGS:\n")
        for i, player in enumerate(results['players'], 1):
            print(f"{i}. {player.name} ({player.position})")
            print(f"   Category: {player.category} | Probability: {player.probability:.1f}%")
            print(f"   Mental State: {player.mental_state} | Fatigue: {player.fatigue_level}")
            print(f"   {player.confidence_explanation}\n")
        
        print("✅ Analysis complete!")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
