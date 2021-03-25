DATA = {
    "summary_csv": "summary.csv",
    "root_dir": "data/"
}

MODEL = {
    "vit_configs": {
        "img_len" : 256,
        "patch_len" : 32, # note that img_len // patch_len needs to be a whole number
        "outpatch_len" : 32,
        "channels": 4,
        "depth" : 1,
        "heads" : 2,
    },   
    "res_block_configs": {
        "block_channels": [(1,1,1)] # e.g. [(3,6,3),(3,5,3)]
    },
    # "rpn_configs" : {}
}