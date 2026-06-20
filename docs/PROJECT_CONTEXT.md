# InfraScan Sentinel - Master Project Context

## Project Overview

InfraScan Sentinel is a civil engineering research project focused on estimating seismic separation gaps between adjacent buildings using ordinary smartphone photographs.

The project is intended as a rapid screening tool rather than a replacement for professional structural inspection.

Primary objective:

Estimate the gap between two neighboring buildings from a single image and compare that estimated gap against seismic code requirements.

Target timeline: 3-4 months.

Target outcome:

* Strong civil engineering research paper
* Structural health monitoring application
* University admissions portfolio project
* Demonstration of engineering methodology rather than just coding ability

---

# Research Question

Can computer vision techniques estimate seismic separation gaps between adjacent buildings using only smartphone photographs?

More specifically:

Can depth estimation and geometric edge detection be combined to identify the inner faces of two buildings and estimate the physical distance between them?

---

# Original Motivation

Many buildings in Nepal are constructed extremely close together.

During earthquakes adjacent buildings can collide due to insufficient seismic separation.

Manual inspection requires field visits and measurements.

A smartphone-based screening system could potentially identify buildings that require further inspection.

The project focuses on:

* Accessibility
* Low cost
* Practical field deployment
* Engineering relevance

---

# High-Level Pipeline

Image

↓

MiDaS depth estimation

↓

Depth map generation

↓

Depth edge extraction

↓

Canny edge detection

↓

Hough line detection

↓

Candidate line generation

↓

Depth discontinuity validation

↓

Valid building-edge selection

↓

Left/right edge pairing

↓

Depth-Hough fusion

↓

Pixel gap estimation

↓

Physical gap estimation

↓

Seismic safety assessment

---

# Current System Architecture

The current system combines two independent sources of information.

1. Hough line geometry

Provides:

* Building edge candidates
* Precise image-space locations
* Strong geometric information

Weakness:

Can detect many irrelevant lines.

Examples:

* Windows
* Shadows
* Architectural details

---

2. MiDaS depth estimation

Provides:

* Relative depth information
* Depth discontinuities
* Building separation cues

Weakness:

Not metrically accurate.

Depth values are relative rather than true distances.

---

# Core Idea

A true building edge should satisfy BOTH:

1. Geometric edge condition

Detected by Hough.

2. Depth discontinuity condition

Detected by MiDaS.

Only lines satisfying both conditions should survive.

This is the central contribution of the current prototype.

---

# Current Functions

## process_image()

Main pipeline controller.

Responsibilities:

* Load image
* Resize image
* Generate depth map
* Detect depth edges
* Run Hough transform
* Generate candidate lines
* Validate lines
* Pair building edges
* Fuse estimates
* Estimate physical gap
* Draw results

---

## detect_candidate_lines()

Purpose:

Convert raw Hough detections into structured candidate lines.

Operations:

* Angle computation
* Vertical filtering
* Length computation
* Perpendicular vector computation
* Storage of geometric information

Output:

length_coordinates_List

Each candidate stores:

* x1
* y1
* x2
* y2
* line length
* dx
* dy
* px
* py
* normalized perpendicular vector

---

## validate_line()

Purpose:

Determine whether a Hough line corresponds to a true depth discontinuity.

Method:

For multiple locations along the line:

* Sample depth on left side
* Sample depth on right side
* Compute depth difference

Produces:

Deltas

Then:

* Compute median delta
* Reject weak discontinuities
* Measure consistency

Returns:

* is_valid
* ratio
* median

This is currently one of the most important functions in the system.

---

## find_best_pair()

Purpose:

Find the most likely pair of building edges.

Method:

1. Split valid lines into left and right groups.
2. Generate possible pairs.
3. Evaluate depth behavior between pairs.
4. Select best candidate pair.

Current strategy:

Smallest valid gap wins.

This is a heuristic and may be improved later.

---

# Important Parameters

## d

Current value approximately:

5

Purpose:

Distance used when sampling depth on both sides of a candidate line.

Larger d:

* More stable
* Less local

Smaller d:

* More local
* More sensitive to noise

---

## N

Current value approximately:

5

Purpose:

Number of samples taken along a line.

Higher values:

* Better robustness
* Slower runtime

---

## threshold_min

Current value approximately:

0.5

Purpose:

Minimum median depth discontinuity.

Rejects weak depth transitions.

---

## tolerance

Current value approximately:

0.3

Purpose:

Controls consistency filtering.

Measures agreement with median depth difference.

---

## w

Typical values:

5
10
20

Purpose:

Neighborhood width used during pair validation.

Strongly influences robustness.

---

# Current Fusion Strategy

The system currently fuses:

Depth estimate

and

Hough estimate

using gap sharpness.

High sharpness:

More trust in depth.

Low sharpness:

More trust in Hough.

This is one of the strongest parts of the current pipeline.

---

# Current Gap Estimation

Pixel gap:

fused_right - fused_left

Physical gap:

(pixel_gap × assumed_distance) / focal_length_pixels

Current assumption:

assumed_distance_cm = 500

This is currently the largest scientific weakness.

---

# Current Strengths

Successfully implemented:

* MiDaS integration
* Depth maps
* Canny edges
* Hough transform
* Candidate filtering
* Depth validation
* Pair selection
* Fusion
* Confidence scoring
* EXIF focal extraction
* Multi-image testing

The system is no longer a simple prototype.

---

# Current Weaknesses

## Weakness 1

Assumed camera distance.

Current:

500 cm

Needs real measurements.

---

## Weakness 2

Limited ground truth.

No large validation dataset yet.

---

## Weakness 3

Mostly tested on frontal photographs.

Perspective effects remain underexplored.

---

## Weakness 4

Pair-selection still heuristic.

Could fail in cluttered scenes.

---

## Weakness 5

MiDaS depth is relative.

Not true metric depth.

---

# Validation Status

Current testing:

Approximately 20-30 real images.

Observed strengths:

* Detects major building edges.
* Fusion often improves results.
* Depth validation removes many false positives.

Observed failures:

* Complex façades.
* Weak depth contrast.
* Strong shadows.
* Severe perspective distortion.

---

# Engineering Lessons Learned

1. Pure Hough detection is insufficient.

2. Pure depth detection is insufficient.

3. Combining both works better.

4. Consistency filtering is important.

5. Frontal photographs perform best.

6. Small parameter changes can noticeably affect performance.

7. Ground truth matters more than adding new features.

---

# Research Roadmap

Phase 1

Architecture cleanup.

Current stage.

Tasks:

* detect_candidate_lines()
* validate_line()
* find_best_pair()

modularization.

---

Phase 2

Ground truth collection.

Tasks:

* Tape-measure measurements
* Known gap measurements
* Controlled image capture

---

Phase 3

Evaluation.

Tasks:

* Error calculation
* Accuracy metrics
* Parameter sensitivity

---

Phase 4

Analysis.

Tasks:

* Failure modes
* Strengths
* Limitations

---

Phase 5

Paper writing.

Sections:

* Introduction
* Literature Review
* Methodology
* Experiments
* Results
* Discussion
* Limitations
* Future Work

---

# Rules For Future AI Assistants

1. Preserve working functionality.

2. Do not add complexity without justification.

3. Prioritize validation over feature additions.

4. Focus on engineering credibility.

5. Focus on completing a publishable project within 4 months.

6. Avoid endless refactoring.

7. Ground truth data is more important than new algorithms.

8. Every major change should improve research quality, not just code quality.

9. Explain reasoning before proposing large modifications.

10. Keep project scope realistic.
