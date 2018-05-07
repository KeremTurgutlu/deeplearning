from feateng import *

def test_reg_mean_encoding():
    """
    test for reg_mean_encoding function
    """
    df = pd.DataFrame({"A":[1,1,1,2,2,2,3],
              "B":[1,1,2,2,3,3,4],
              "y":[1,0,1,0,0,1,1]})
    
    # single column test
    out_df = reg_mean_encoding(df, "A", "encode", "y", splits=2, seed=10)
    true = [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.25]
    assert true == list(out_df["encode"].values)
    
    # multi column test
    out_df = reg_mean_encoding(df, ["A", "B"], "encode", "y", splits=2, seed=10)
    true = [0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert true == list(out_df["encode"].values)
    
    
    
def test_reg_count_encoding():
    """
    test for reg_mean_encoding function
    """
    df = pd.DataFrame({"A":[1,1,1,2,2,2,3],
              "B":[1,1,2,2,3,3,4],
              "y":[1,0,1,0,0,1,1]})
    
    # single column test
    out_df = reg_count_encoding(df, "A", "encode", "y", splits=2, seed=10)
    true = [1,2,1,2,1,1,1]
    assert true == list(out_df["encode"].values)
    
    # multi column test
    out_df = reg_count_encoding(df, ["A", "B"], "encode", "y", splits=2, seed=10)
    true = [1,1,1,1,1,1,1]
    assert true == list(out_df["encode"].values)