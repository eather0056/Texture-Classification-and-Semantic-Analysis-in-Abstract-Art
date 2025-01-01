now next task corelation between texture, emmotion and images



eth@eth:~/Texture-Classification-and-Semantic-Analysis-in-Abstract-Art$ python3 src/Correlation_Analysis.py 
Combined DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21550 entries, 0 to 21549
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   filename      21550 non-null  object 
 1   texture       21550 non-null  object 
 2   category_x    21550 non-null  object 
 3   emotion       21550 non-null  object 
 4   category_y    21550 non-null  object 
 5   rationale     21550 non-null  object 
 6   polarity      21550 non-null  float64
 7   subjectivity  21550 non-null  float64
 8   annotator     21550 non-null  object 
dtypes: float64(2), object(7)
memory usage: 1.5+ MB
None
Sample Combined Data:
                                            filename texture category_x emotion category_y                                          rationale  polarity  subjectivity      annotator
0  edward-avedisian_untitled-203-watercolor-ball-...  smooth     smooth   anger        MIN  I've seen this same image in multiple HITs, an...     0.230         0.380  annotator_260
1  edward-avedisian_untitled-203-watercolor-ball-...  smooth     smooth   anger        MIN      The pink and yellow contrast each other well.    -0.075         0.225  annotator_285
2  edward-avedisian_untitled-203-watercolor-ball-...  smooth     smooth   anger        MIN  Reminds me of a brightly colored piece of cand...     0.700         0.800  annotator_260
3  edward-avedisian_untitled-203-watercolor-ball-...  smooth     smooth   anger        MIN                        bright color so better feel     0.600         0.650  annotator_280
4  edward-avedisian_untitled-203-watercolor-ball-...  smooth     smooth   anger        MIN                     image is bright and simplistic     0.100         0.650   annotator_32
Grouped Data:
   texture    emotion  polarity  subjectivity  image_count
0  chaotic      anger -0.023698      0.385857          871
1  chaotic    disgust -0.002612      0.366143         2035
2  chaotic       fear -0.000640      0.366749         1768
3  chaotic  happiness  0.132919      0.396162         1175
4  chaotic    sadness -0.035212      0.383472          478
Grouped data saved to data/processed/grouped_texture_emotion.csv
Correlation Matrix Input Data:
   polarity  subjectivity  image_count  texture_chaotic  texture_circular  texture_dots  texture_lines  texture_rough  texture_smooth
0     0.230         0.380          367            False             False         False          False          False            True
1    -0.075         0.225          367            False             False         False          False          False            True
2     0.700         0.800          367            False             False         False          False          False            True
3     0.600         0.650          367            False             False         False          False          False            True
4     0.100         0.650          367            False             False         False          False          False            True
Correlation Matrix:
                  polarity  subjectivity  image_count  texture_chaotic  texture_circular  texture_dots  texture_lines  texture_rough  texture_smooth
polarity          1.000000      0.157380    -0.043558        -0.025824         -0.001095      0.010936       0.022394      -0.021367        0.034410
subjectivity      0.157380      1.000000    -0.010746         0.001889         -0.001015     -0.005848      -0.002449       0.011437       -0.011750
image_count      -0.043558     -0.010746     1.000000         0.298280         -0.390791     -0.218605      -0.328823       0.406499       -0.237809
texture_chaotic  -0.025824      0.001889     0.298280         1.000000         -0.169152     -0.075558      -0.262066      -0.439707       -0.290878
texture_circular -0.001095     -0.001015    -0.390791        -0.169152          1.000000     -0.030751      -0.106657      -0.178955       -0.118383
texture_dots      0.010936     -0.005848    -0.218605        -0.075558         -0.030751      1.000000      -0.047642      -0.079936       -0.052880
texture_lines     0.022394     -0.002449    -0.328823        -0.262066         -0.106657     -0.047642       1.000000      -0.277253       -0.183410
texture_rough    -0.021367      0.011437     0.406499        -0.439707         -0.178955     -0.079936      -0.277253       1.000000       -0.307735
texture_smooth    0.034410     -0.011750    -0.237809        -0.290878         -0.118383     -0.052880      -0.183410      -0.307735        1.000000
Correlation matrix saved to data/processed/correlation_matrix.csv