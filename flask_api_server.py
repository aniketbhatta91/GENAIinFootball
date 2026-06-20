"""
Flask API Server for Penalty Shootout Predictor
Serves the React frontend and provides REST endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from penalty_predictor_backend import (
    PenaltyShootoutAnalyzer, 
    PredictionResult
)
import json
from dataclasses import asdict
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analyzer
analyzer = PenaltyShootoutAnalyzer()

# ────────────────────────────────────────────────────────────────
# API Endpoints
# ────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Penalty Shootout Analyzer',
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_commentary():
    """
    Main analysis endpoint
    POST body: { "commentary": "match commentary text" }
    Returns: player classifications and predictions
    """
    try:
        data = request.get_json()
        
        if not data or 'commentary' not in data:
            return jsonify({'error': 'Missing commentary field'}), 400
        
        commentary = data['commentary'].strip()
        
        if len(commentary) < 20:
            return jsonify({
                'error': 'Commentary too short (minimum 20 characters)',
                'received_length': len(commentary)
            }), 400
        
        logger.info(f"Analyzing commentary ({len(commentary)} chars)")
        
        # Run analysis
        results = analyzer.analyze(commentary)
        
        # Format response
        response = {
            'success': True,
            'timestamp': results['timestamp'],
            'match_context': results['match_context'],
            'players': [
                {
                    'rank': i + 1,
                    'name': p.name,
                    'position': p.position,
                    'category': p.category,
                    'probability': round(p.probability, 1),
                    'mental_state': p.mental_state,
                    'fatigue_level': p.fatigue_level,
                    'stress_count': p.stress_count,
                    'confidence_explanation': p.confidence_explanation
                }
                for i, p in enumerate(results['players'])
            ],
            'summary': results['summary'],
            'statistics': {
                'total_players_analyzed': len(results['players']),
                'strong_candidates': sum(1 for p in results['players'] if p.category == 'STRONG'),
                'moderate_candidates': sum(1 for p in results['players'] if p.category == 'MODERATE'),
                'weak_candidates': sum(1 for p in results['players'] if p.category == 'WEAK')
            }
        }
        
        return jsonify(response), 200
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Batch analysis endpoint for multiple matches
    POST body: { "matches": [ { "id": "match1", "commentary": "..." }, ... ] }
    """
    try:
        data = request.get_json()
        
        if not data or 'matches' not in data:
            return jsonify({'error': 'Missing matches field'}), 400
        
        matches = data['matches']
        if not isinstance(matches, list) or len(matches) == 0:
            return jsonify({'error': 'matches must be a non-empty list'}), 400
        
        results = []
        errors = []
        
        for match in matches:
            try:
                match_id = match.get('id', f'match_{len(results)}')
                commentary = match.get('commentary', '').strip()
                
                if len(commentary) < 20:
                    errors.append({
                        'match_id': match_id,
                        'error': 'Commentary too short'
                    })
                    continue
                
                analysis = analyzer.analyze(commentary)
                
                results.append({
                    'match_id': match_id,
                    'success': True,
                    'players': [
                        {
                            'name': p.name,
                            'category': p.category,
                            'probability': round(p.probability, 1)
                        }
                        for p in analysis['players']
                    ]
                })
            except Exception as e:
                errors.append({
                    'match_id': match.get('id', 'unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'success': len(results) > 0,
            'results': results,
            'errors': errors,
            'processed': len(results),
            'failed': len(errors)
        }), 200
    
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example commentaries for testing"""
    examples = [
        {
            'title': 'Tight Match (Extra Time)',
            'description': 'Close game with high pressure in final 15 minutes',
            'commentary': """Min 76: Kane drives forward but his pass is intercepted. He looks frustrated with himself.
Min 77: The pressure is immense on both teams. Players are beginning to tire.
Min 78: Sterling with a brilliant run down the left wing, creates space and nearly scores! Confident performance.
Min 79: Mount controls the midfield beautifully, distributing passes with calm precision despite the intense pressure.
Min 80: Saka looks exhausted after a grueling 80 minutes. Has lost the ball twice in quick succession, looks demoralized.
Min 82: Kane with a clinical chance! The goalkeeper makes a world-class save. Kane is frustrated but composed.
Min 83: Mount maintains his composure under extreme pressure. Fresh legs and clear thinking evident.
Min 84: Saka gives away possession carelessly. Clearly running out of energy, making poor decisions.
Min 85: Sterling bursts forward with renewed energy! Nearly creates a breakthrough moment for his team."""
        },
        {
            'title': 'Dominant Performance',
            'description': 'Team controlling the match with confident players',
            'commentary': """Min 72: Harry Maguire heads clear decisively. No panic in the defense.
Min 74: Bruno Fernandes controls the tempo, slows the game down perfectly. Composed leadership.
Min 76: Rashford charges down the wing with confidence. Quick feet, aggressive but controlled.
Min 78: The team is dominating. Possession is secure. Players look fresh and determined.
Min 80: Cavani positions himself well in the box. Waiting for the perfect moment, clinical finisher.
Min 82: Fred wins the ball back cleanly. His positioning is excellent, energy remains high.
Min 84: Malacia makes a perfect tackle, recovers quickly. Alert and focused.
Min 85: The team looks in control. No visible signs of fatigue. High morale evident."""
        }
    ]
    
    return jsonify({
        'examples': examples,
        'count': len(examples)
    })

@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Get API request/response schema"""
    return jsonify({
        'request': {
            'method': 'POST',
            'endpoint': '/api/analyze',
            'body': {
                'commentary': 'string (minimum 20 characters)',
                'example': 'Min 76: Kane looks frustrated...'
            }
        },
        'response': {
            'success': 'boolean',
            'timestamp': 'ISO 8601 string',
            'match_context': {
                'overall_intensity': 'HIGH|MEDIUM|LOW',
                'pressure_level': 'HIGH|MEDIUM|LOW',
                'team_morale': 'POSITIVE|NEUTRAL|NEGATIVE'
            },
            'players': [
                {
                    'rank': 'integer',
                    'name': 'string',
                    'position': 'string',
                    'category': 'STRONG|MODERATE|WEAK',
                    'probability': 'float (0-100)',
                    'mental_state': 'POSITIVE|NEUTRAL|NEGATIVE',
                    'fatigue_level': 'LOW|MEDIUM|HIGH'
                }
            ],
            'statistics': {
                'total_players_analyzed': 'integer',
                'strong_candidates': 'integer',
                'moderate_candidates': 'integer',
                'weak_candidates': 'integer'
            }
        },
        'categories': {
            'STRONG': '70%+ probability. Positive sentiment, fresh, confident.',
            'MODERATE': '40-70% probability. Mixed signals, average condition.',
            'WEAK': '<40% probability. Negative sentiment, tired, poor form.'
        }
    })

# ────────────────────────────────────────────────────────────────
# Error Handlers
# ────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ────────────────────────────────────────────────────────────────
# Startup
# ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         PENALTY SHOOTOUT ANALYZER API SERVER              ║
    ║                   v1.0.0                                  ║
    ╚════════════════════════════════════════════════════════════╝
    
    🚀 Server starting...
    
    📍 Endpoints:
       GET  /api/health          - Health check
       POST /api/analyze         - Analyze single match
       POST /api/analyze/batch   - Analyze multiple matches
       GET  /api/examples        - Get example commentaries
       GET  /api/schema          - Get API schema
    
    🌐 Frontend:
       http://localhost:5000
    
    📚 API Documentation:
       http://localhost:5000/api/schema
    """)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
