"""
Flask API Server for Penalty Shootout Predictor - OFFLINE VERSION
NO BERT DOWNLOAD - Works immediately
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from penalty_predictor_backend_OFFLINE import PenaltyShootoutAnalyzer
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analyzer
print("\n✅ Starting Penalty Shootout Analyzer (Offline Mode)")
analyzer = PenaltyShootoutAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Penalty Shootout Analyzer',
        'version': '1.0.0',
        'mode': 'OFFLINE (No BERT required)'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_commentary():
    """Main analysis endpoint"""
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
        
        results = analyzer.analyze(commentary)
        
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
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example commentaries"""
    examples = [
        {
            'title': 'Tight Match (Extra Time)',
            'description': 'Close game with high pressure',
            'commentary': """Min 76: Kane drives forward but his pass is intercepted. He looks frustrated.
Min 77: The pressure is immense on both teams. Players are beginning to tire.
Min 78: Sterling with a brilliant run down the wing! Confident performance.
Min 79: Mount controls the midfield beautifully, calm precision despite pressure.
Min 80: Saka looks exhausted after 80 minutes. Lost the ball twice.
Min 82: Kane with a clinical chance!
Min 83: Mount maintains composure. Fresh legs showing.
Min 84: Saka gives away possession carelessly. Worn out.
Min 85: Sterling bursts forward with energy!"""
        }
    ]
    
    return jsonify({'examples': examples, 'count': len(examples)})

@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Get API schema"""
    return jsonify({
        'request': {
            'method': 'POST',
            'endpoint': '/api/analyze',
            'body': {'commentary': 'string (minimum 20 characters)'}
        },
        'response': {
            'success': 'boolean',
            'timestamp': 'ISO 8601 string',
            'players': [
                {
                    'rank': 'integer',
                    'name': 'string',
                    'category': 'STRONG|MODERATE|WEAK',
                    'probability': 'float (0-100)'
                }
            ]
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         PENALTY SHOOTOUT ANALYZER API SERVER              ║
    ║              OFFLINE MODE (No BERT Download)              ║
    ║                      v1.0.0                               ║
    ╚════════════════════════════════════════════════════════════╝
    
    🚀 Server starting...
    
    📍 Endpoints:
       GET  /api/health          - Health check
       POST /api/analyze         - Analyze single match
       GET  /api/examples        - Get example commentaries
       GET  /api/schema          - Get API schema
    
    🌐 Frontend:
       http://localhost:5000
    
    ⚡ Status: OFFLINE MODE (No internet needed for analysis)
    """)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
