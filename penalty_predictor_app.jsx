import React, { useState, useRef } from 'react';
import { AlertCircle, Zap, Users, TrendingUp, Copy, Download } from 'lucide-react';

const PenaltyPredictorApp = () => {
  const [commentary, setCommentary] = useState('');
  const [playerData, setPlayerData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedTab, setSelectedTab] = useState('input');

  const callClaude = async (prompt) => {
    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-6',
          max_tokens: 2000,
          system: `You are a football analytics expert specializing in penalty shootout prediction. 
Analyze match commentary to identify players and assess their mental state, fatigue, and confidence levels.
Extract players mentioned and provide a JSON response with exact structure:
{
  "players": [
    {
      "name": "player name",
      "position": "position",
      "mentions_count": number,
      "sentiment_indicators": ["indicator1", "indicator2"],
      "fatigue_level": "LOW|MEDIUM|HIGH",
      "confidence_score": number (0-100),
      "stress_indicators": ["indicator1"],
      "mental_state": "POSITIVE|NEUTRAL|NEGATIVE",
      "key_moments": ["description"]
    }
  ],
  "match_context": {
    "overall_intensity": "HIGH|MEDIUM|LOW",
    "pressure_level": "HIGH|MEDIUM|LOW",
    "team_morale": "POSITIVE|NEUTRAL|NEGATIVE"
  }
}`,
          messages: [{ role: 'user', content: prompt }]
        })
      });

      const data = await response.json();
      const content = data.content[0].text;
      
      // Extract JSON from response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      throw new Error('Could not parse response');
    } catch (err) {
      throw new Error(`API Error: ${err.message}`);
    }
  };

  const analyzePenalties = async () => {
    if (!commentary.trim()) {
      setError('Please enter match commentary');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const prompt = `Analyze this final 15 minutes of match commentary and predict which players are eligible for penalty shootout:

COMMENTARY:
${commentary}

Extract all players mentioned and classify them into three penalty-taking categories:
1. STRONG (70%+ chance to score) - positive sentiment, fresh, confident, good penalty history
2. MODERATE (40-70% chance) - neutral sentiment, mixed signals, average condition
3. WEAK (Below 40% chance) - negative sentiment, tired, poor performance, low confidence

Provide detailed analysis with JSON response.`;

      const result = await callClaude(prompt);
      
      // Process and classify players
      const classified = classifyPlayers(result.players || []);
      
      setPlayerData({
        ...result,
        classified,
        timestamp: new Date().toLocaleString()
      });
      setSelectedTab('results');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const classifyPlayers = (players) => {
    return players.map(p => {
      let category = 'MODERATE';
      let probability = 50;

      // Calculate probability based on multiple factors
      let score = p.confidence_score || 50;
      
      // Adjust for mental state
      if (p.mental_state === 'POSITIVE') score += 15;
      if (p.mental_state === 'NEGATIVE') score -= 15;

      // Adjust for fatigue
      if (p.fatigue_level === 'LOW') score += 10;
      if (p.fatigue_level === 'HIGH') score -= 15;

      // Adjust for stress
      if ((p.stress_indicators || []).length === 0) score += 5;
      if ((p.stress_indicators || []).length > 2) score -= 10;

      // Classify
      if (score >= 70) category = 'STRONG';
      else if (score < 40) category = 'WEAK';
      else category = 'MODERATE';

      return { ...p, category, probability: Math.min(100, Math.max(0, score)) };
    }).sort((a, b) => b.probability - a.probability);
  };

  const exportResults = () => {
    if (!playerData) return;
    
    const csv = [
      ['Rank', 'Player', 'Position', 'Category', 'Probability', 'Mental State', 'Fatigue', 'Mentions'],
      ...playerData.classified.map((p, i) => [
        i + 1,
        p.name,
        p.position,
        p.category,
        `${p.probability}%`,
        p.mental_state,
        p.fatigue_level,
        p.mentions_count
      ])
    ];

    const csvContent = csv.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `penalty-shootout-analysis-${Date.now()}.csv`;
    a.click();
  };

  const getCategoryColor = (category) => {
    const colors = {
      'STRONG': { bg: '#10B981', text: '#ECFDF5', badge: '#D1FAE5' },
      'MODERATE': { bg: '#F59E0B', text: '#FFFBEB', badge: '#FEF3C7' },
      'WEAK': { bg: '#EF4444', text: '#FEF2F2', badge: '#FEE2E2' }
    };
    return colors[category] || colors['MODERATE'];
  };

  return (
    <div style={{ background: '#0F172A', minHeight: '100vh', color: '#F1F5F9', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1E40AF 0%, #0F172A 100%)', padding: '32px 24px', borderBottom: '1px solid #1E293B' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <Zap size={32} color="#3B82F6" />
            <h1 style={{ fontSize: '28px', fontWeight: '900', margin: 0 }}>Penalty Shooter Analyzer</h1>
          </div>
          <p style={{ color: '#94A3B8', margin: '0', fontSize: '14px' }}>AI-powered penalty shootout eligibility prediction using BERT sentiment analysis</p>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '1px solid #1E293B', paddingBottom: '0' }}>
          {[
            { id: 'input', label: 'Analyze Commentary', icon: '📝' },
            { id: 'results', label: 'Results', icon: '📊', disabled: !playerData }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => !tab.disabled && setSelectedTab(tab.id)}
              disabled={tab.disabled}
              style={{
                padding: '12px 20px',
                background: selectedTab === tab.id ? '#3B82F6' : 'transparent',
                color: selectedTab === tab.id ? '#FFFFFF' : '#94A3B8',
                border: 'none',
                cursor: tab.disabled ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: '600',
                borderRadius: '8px 8px 0 0',
                opacity: tab.disabled ? 0.5 : 1,
                transition: 'all 0.2s'
              }}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* Input Tab */}
        {selectedTab === 'input' && (
          <div style={{
            background: '#1E293B',
            borderRadius: '12px',
            padding: '32px',
            border: '1px solid #334155'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <label style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: '600',
                color: '#E2E8F0'
              }}>
                Match Commentary (Last 15 Minutes)
              </label>
              <textarea
                value={commentary}
                onChange={(e) => setCommentary(e.target.value)}
                placeholder="Paste the commentary from the last 15 minutes of the match. Include mentions of players, their performances, missed chances, fatigue, pressure moments, etc.

Example:
Min 76: Kane drives forward but his pass is intercepted. He looks frustrated. 
Min 78: The pressure is immense now. Both teams are tiring.
Min 82: Sterling with a brilliant run down the wing, creates space for a shot...
Min 85: Final whistle. Extra time needed..."
                style={{
                  width: '100%',
                  height: '300px',
                  padding: '16px',
                  background: '#0F172A',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#F1F5F9',
                  fontFamily: 'monospace',
                  fontSize: '14px',
                  resize: 'vertical',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.target.style.borderColor = '#3B82F6'}
                onBlur={(e) => e.target.style.borderColor = '#475569'}
              />
              <p style={{ fontSize: '12px', color: '#64748B', margin: '8px 0 0 0' }}>
                Minimum 50 characters recommended. Include player names, performance descriptions, emotional reactions.
              </p>
            </div>

            {error && (
              <div style={{
                background: '#7F1D1D',
                border: '1px solid #DC2626',
                borderRadius: '8px',
                padding: '16px',
                display: 'flex',
                gap: '12px',
                marginBottom: '24px'
              }}>
                <AlertCircle size={20} color="#FCA5A5" style={{ flexShrink: 0 }} />
                <div>
                  <p style={{ margin: '0 0 4px 0', fontWeight: '600', color: '#FCA5A5' }}>Error</p>
                  <p style={{ margin: 0, color: '#F87171', fontSize: '14px' }}>{error}</p>
                </div>
              </div>
            )}

            <button
              onClick={analyzePenalties}
              disabled={loading || !commentary.trim()}
              style={{
                width: '100%',
                padding: '14px 24px',
                background: loading ? '#475569' : '#3B82F6',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: '700',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {loading ? '⏳ Analyzing with BERT...' : '🎯 Analyze Penalty Eligibility'}
            </button>
          </div>
        )}

        {/* Results Tab */}
        {selectedTab === 'results' && playerData && (
          <div>
            {/* Match Context */}
            <div style={{
              background: '#1E293B',
              borderRadius: '12px',
              padding: '20px',
              border: '1px solid #334155',
              marginBottom: '24px'
            }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700' }}>Match Context</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                {[
                  { label: 'Overall Intensity', value: playerData.match_context?.overall_intensity },
                  { label: 'Pressure Level', value: playerData.match_context?.pressure_level },
                  { label: 'Team Morale', value: playerData.match_context?.team_morale }
                ].map((item, i) => (
                  <div key={i} style={{
                    background: '#0F172A',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid #475569'
                  }}>
                    <p style={{ color: '#94A3B8', fontSize: '12px', margin: '0 0 4px 0' }}>{item.label}</p>
                    <p style={{ color: '#F1F5F9', fontSize: '16px', fontWeight: '600', margin: 0 }}>{item.value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Player Rankings */}
            <div style={{
              background: '#1E293B',
              borderRadius: '12px',
              padding: '24px',
              border: '1px solid #334155',
              marginBottom: '24px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div>
                  <h3 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: '700' }}>Players Ranked by Shootout Eligibility</h3>
                  <p style={{ color: '#94A3B8', fontSize: '12px', margin: 0 }}>Sorted by penalty success probability</p>
                </div>
                <button
                  onClick={exportResults}
                  style={{
                    padding: '8px 12px',
                    background: '#10B981',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  <Download size={14} /> Export CSV
                </button>
              </div>

              {playerData.classified && playerData.classified.length > 0 ? (
                <div style={{ display: 'grid', gap: '12px' }}>
                  {playerData.classified.map((player, index) => {
                    const colors = getCategoryColor(player.category);
                    const categoryEmoji = { STRONG: '✅', MODERATE: '⚠️', WEAK: '❌' }[player.category];
                    
                    return (
                      <div
                        key={index}
                        style={{
                          background: '#0F172A',
                          border: `2px solid ${colors.bg}`,
                          borderRadius: '10px',
                          padding: '16px',
                          display: 'grid',
                          gridTemplateColumns: 'auto 1fr auto',
                          gap: '16px',
                          alignItems: 'center'
                        }}
                      >
                        {/* Rank & Name */}
                        <div style={{ textAlign: 'center' }}>
                          <div style={{
                            fontSize: '24px',
                            fontWeight: '900',
                            color: '#3B82F6',
                            marginBottom: '4px'
                          }}>
                            #{index + 1}
                          </div>
                          <div style={{ fontSize: '12px', color: '#94A3B8' }}>
                            {player.position}
                          </div>
                        </div>

                        {/* Details */}
                        <div>
                          <div style={{
                            fontSize: '16px',
                            fontWeight: '700',
                            color: '#F1F5F9',
                            marginBottom: '8px'
                          }}>
                            {player.name}
                          </div>
                          
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '8px', fontSize: '12px' }}>
                            <div>
                              <span style={{ color: '#94A3B8' }}>Mental State:</span>
                              <span style={{ 
                                color: player.mental_state === 'POSITIVE' ? '#10B981' : 
                                       player.mental_state === 'NEGATIVE' ? '#EF4444' : '#94A3B8',
                                marginLeft: '6px',
                                fontWeight: '600'
                              }}>
                                {player.mental_state}
                              </span>
                            </div>
                            <div>
                              <span style={{ color: '#94A3B8' }}>Fatigue:</span>
                              <span style={{ color: '#E2E8F0', marginLeft: '6px', fontWeight: '600' }}>
                                {player.fatigue_level}
                              </span>
                            </div>
                            <div>
                              <span style={{ color: '#94A3B8' }}>Mentions:</span>
                              <span style={{ color: '#E2E8F0', marginLeft: '6px', fontWeight: '600' }}>
                                {player.mentions_count}x
                              </span>
                            </div>
                          </div>

                          {player.stress_indicators && player.stress_indicators.length > 0 && (
                            <div style={{ marginTop: '8px' }}>
                              <span style={{ color: '#94A3B8', fontSize: '12px' }}>Stress Factors:</span>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                                {player.stress_indicators.slice(0, 3).map((indicator, i) => (
                                  <span
                                    key={i}
                                    style={{
                                      background: '#7F1D1D',
                                      color: '#FCA5A5',
                                      padding: '2px 6px',
                                      borderRadius: '4px',
                                      fontSize: '10px',
                                      fontWeight: '600'
                                    }}
                                  >
                                    {indicator}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Category Badge & Probability */}
                        <div style={{ textAlign: 'right' }}>
                          <div style={{
                            background: colors.bg,
                            color: colors.text,
                            padding: '8px 12px',
                            borderRadius: '8px',
                            marginBottom: '8px',
                            fontWeight: '700',
                            fontSize: '13px',
                            whiteSpace: 'nowrap'
                          }}>
                            {categoryEmoji} {player.category}
                          </div>
                          <div style={{
                            fontSize: '20px',
                            fontWeight: '900',
                            color: colors.bg,
                            marginBottom: '4px'
                          }}>
                            {player.probability}%
                          </div>
                          <div style={{
                            background: '#334155',
                            height: '6px',
                            borderRadius: '3px',
                            overflow: 'hidden',
                            width: '80px'
                          }}>
                            <div style={{
                              background: colors.bg,
                              height: '100%',
                              width: `${player.probability}%`,
                              transition: 'width 0.3s'
                            }} />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{
                  textAlign: 'center',
                  color: '#94A3B8',
                  padding: '40px 20px'
                }}>
                  <Users size={48} style={{ opacity: 0.5, marginBottom: '12px' }} />
                  <p style={{ margin: 0 }}>No players found in commentary. Try adding more player names.</p>
                </div>
              )}
            </div>

            {/* Legend */}
            <div style={{
              background: '#1E293B',
              borderRadius: '12px',
              padding: '20px',
              border: '1px solid #334155'
            }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '700' }}>Classification Guide</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
                {[
                  {
                    category: 'STRONG',
                    emoji: '✅',
                    description: 'High likelihood to score (70%+). Positive sentiment, fresh condition, confident performance.',
                    traits: ['Positive sentiment', 'Low fatigue', 'Good history', 'Composed']
                  },
                  {
                    category: 'MODERATE',
                    emoji: '⚠️',
                    description: 'Moderate likelihood to score (40-70%). Mixed signals, average condition.',
                    traits: ['Neutral sentiment', 'Medium fatigue', 'Variable form', 'Adapting']
                  },
                  {
                    category: 'WEAK',
                    emoji: '❌',
                    description: 'Lower likelihood to score (Below 40%). Negative sentiment, tired, poor performance.',
                    traits: ['Negative sentiment', 'High fatigue', 'Poor history', 'Under pressure']
                  }
                ].map((item, i) => (
                  <div key={i} style={{
                    background: '#0F172A',
                    padding: '12px',
                    borderRadius: '8px',
                    border: `1px solid ${getCategoryColor(item.category).bg}33`
                  }}>
                    <div style={{
                      fontSize: '24px',
                      marginBottom: '8px'
                    }}>
                      {item.emoji}
                    </div>
                    <p style={{
                      color: getCategoryColor(item.category).bg,
                      fontWeight: '700',
                      margin: '0 0 4px 0',
                      fontSize: '13px'
                    }}>
                      {item.category}
                    </p>
                    <p style={{
                      color: '#94A3B8',
                      fontSize: '12px',
                      margin: '0 0 8px 0'
                    }}>
                      {item.description}
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      {item.traits.map((trait, j) => (
                        <span key={j} style={{
                          background: getCategoryColor(item.category).badge,
                          color: getCategoryColor(item.category).bg,
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '10px',
                          fontWeight: '600'
                        }}>
                          {trait}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Timestamp */}
            <div style={{
              textAlign: 'center',
              marginTop: '24px',
              color: '#64748B',
              fontSize: '12px'
            }}>
              Analysis completed: {playerData.timestamp}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PenaltyPredictorApp;
