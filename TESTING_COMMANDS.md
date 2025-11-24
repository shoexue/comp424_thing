# Testing Your Student Agent

## Quick Test Commands

### 1. Test Against Random Agent

**Single game (no visualization):**
```bash
python simulator.py --player_1 student_agent --player_2 random_agent
```

**Single game (with visualization):**
```bash
python simulator.py --player_1 student_agent --player_2 random_agent --display
```

**Autoplay 100 games (official test format):**
```bash
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay
```

**Test as Player 2 (swap sides):**
```bash
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```

### 2. Test Against Greedy Corners Agent

**Single game:**
```bash
python simulator.py --player_1 student_agent --player_2 greedy_corners_agent
```

**With visualization:**
```bash
python simulator.py --player_1 student_agent --player_2 greedy_corners_agent --display
```

**Autoplay 100 games:**
```bash
python simulator.py --player_1 student_agent --player_2 greedy_corners_agent --autoplay
```

**Test as Player 2:**
```bash
python simulator.py --player_1 greedy_corners_agent --player_2 student_agent --autoplay
```

### 3. Test Against Yourself (Both Sides)

**See how your agent plays against itself:**
```bash
python simulator.py --player_1 student_agent --player_2 student_agent --display
```

### 4. Test on Specific Board

**Test on a specific board file:**
```bash
python simulator.py --player_1 student_agent --player_2 random_agent --board_path boards/empty_7x7.csv --display
```

**Available boards:**
- `boards/empty_7x7.csv`
- `boards/big_x.csv`
- `boards/plus1.csv`
- `boards/plus2.csv`
- `boards/point4.csv`
- `boards/the_circle.csv`
- `boards/the_wall.csv`
- `boards/watch_the_sides.csv`

### 5. Test as Human Player

**Play against your agent:**
```bash
python simulator.py --player_1 human_agent --player_2 student_agent --display
```

**Your agent plays first:**
```bash
python simulator.py --player_1 student_agent --player_2 human_agent --display
```

## Recommended Testing Workflow

### Step 1: Quick Visual Test
```bash
python simulator.py --player_1 student_agent --player_2 random_agent --display
```
- Watch a few games to see if moves look reasonable
- Check that moves are valid and agent doesn't crash

### Step 2: Performance Test vs Random
```bash
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay
```
- Should achieve >80% win rate against random
- Check the win percentage in the output

### Step 3: Performance Test vs Greedy Corners
```bash
python simulator.py --player_1 student_agent --player_2 greedy_corners_agent --autoplay
```
- This is a stronger opponent
- Aim for >50% win rate (beating a greedy agent is good)

### Step 4: Test Both Sides
```bash
# As Player 1
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay

# As Player 2
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```
- Your agent should work well regardless of which side it plays

### Step 5: Final Verification (Official Test Format)
```bash
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```
- This is the exact command that will be used for grading
- Make sure it works perfectly!

## Understanding the Output

### Single Game Output
You'll see:
- Move-by-move logging
- Final scores
- Time taken per move

### Autoplay Output
You'll see:
```
Player 1, agent student_agent, win percentage: 0.85. Maximum turn time was 1.234 seconds.
Player 2, agent random_agent, win percentage: 0.15. Maximum turn time was 0.012 seconds.
```

**Key metrics:**
- **Win percentage**: Should be >0.5 (50%) to be winning
- **Maximum turn time**: Should be <2 seconds (ideally <1.8s with buffer)

## Troubleshooting

### Agent crashes or returns invalid moves
- Check that you're returning a `MoveCoordinates` object
- Verify moves are valid using `check_move_validity()`
- Test with `--display` to see what's happening

### Takes too long (>2 seconds)
- Add time checking in your search
- Break early when approaching time limit
- Test on largest boards (size 12)

### Low win rate
- Improve your evaluation function
- Add lookahead (minimax)
- Test different heuristics

## Example Testing Session

```bash
# 1. Quick visual check
python simulator.py --player_1 student_agent --player_2 random_agent --display

# 2. Performance baseline
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay

# 3. Test against stronger opponent
python simulator.py --player_1 student_agent --player_2 greedy_corners_agent --autoplay

# 4. Final check (official format)
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```

