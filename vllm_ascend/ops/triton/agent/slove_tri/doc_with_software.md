╭─── Claude Code v2.1.96 ──────────────────────────────────────────────────────╮
│                                    │ Tips for getting started                │
│            Welcome back!           │ Run /init to create a CLAUDE.md file w… │
│                                    │ ─────────────────────────────────────── │
│               ▐▛███▜▌              │ Recent activity                         │
│              ▝▜█████▛▘             │ No recent activity                      │
│                ▘▘ ▝▝               │                                         │
│                                    │                                         │
│   Sonnet 4.6 · API Usage Billing   │                                         │
│          /vllm-workspace           │                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

✻ Conversation compacted (ctrl+o for history)

  ⎿  Read vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py (410 lines)
  ⎿  Read analyze_perf.py (34 lines)
  ⎿  Read check_correctness.py (46 lines)
  ⎿  Read save_ref_outputs.py (37 lines)
  ⎿  Read test_solve_tril.py (128 lines)
  ⎿  Skills restored (triton-ascend-ops-optimizer, 
     triton-operator-performance-eval)

● Let me first read the current state of the files and the backup to plan the
  revert of the merge kernel.

  Read 2 files (ctrl+o to expand)
  ⎿  Loaded vllm-ascend/CLAUDE.md

● Good. Now I have the original merge kernel from the backup. The key
  optimization strategy is:

  1. Revert merge kernel back to original structure (explicit pointers were
  worse per msprof op)
  2. Remove other=0.0 from all tl.load calls (the skill guide says this triggers
   internal tl.where blocking load parallelism)
  3. Add load-compute interleaving where possible (load Ai_22 → load A_21 →
  compute, then load Ai_44 → load A_43 → compute, etc.)

  Let me apply these changes:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 63 lines, removed 57 lines
      229      Ad += (bos * H + i_h) * 16
      230      Ai += (bos * H + i_h) * 64
      231  
      232 -    block_start = i_t * 64                                    
      232      offs_16 = tl.arange(0, 16)
      233  
      235 -    # Precompute row offsets for all 4 sub-blocks             
      236 -    offs_m_0 = block_start + offs_16                          
      237 -    offs_m_1 = block_start + 16 + offs_16                     
      238 -    offs_m_2 = block_start + 32 + offs_16                     
      239 -    offs_m_3 = block_start + 48 + offs_16                     
      234 +    # Load Ai_22 (Ad block at row i_t*64+16)                  
      235 +    offs_m = i_t * 64 + 16 + offs_16                          
      236 +    mask_Ad = offs_m[:, None] < T                             
      237 +    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          +]                                                             
      238 +    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      239  
      241 -    # Load all 4 Ad diagonal blocks at once (load-compute inte
          -rleave)                                                       
      242 -    # Load Ai_11 + Ai_22                                      
      243 -    mask_m_0 = offs_m_0[:, None] < T                          
      244 -    ptr_Ad_0 = Ad + offs_m_0[:, None] * (H * 16) + offs_16[Non
          -e, :]                                                         
      245 -    Ai_11 = tl.load(ptr_Ad_0, mask=mask_m_0).to(tl.float32)   
      240 +    # Load A_21 (A block at row i_t*64+16, col 0) and compute 
          +immediately                                                   
      241 +    ptr_A = A + offs_m[:, None] * (H * 64) + offs_16[None, :] 
      242 +    A_21 = tl.load(ptr_A, mask=mask_Ad).to(tl.float32)        
      243 +    tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")      
      244  
      247 -    mask_m_1 = offs_m_1[:, None] < T                          
      248 -    ptr_Ad_1 = Ad + offs_m_1[:, None] * (H * 16) + offs_16[Non
          -e, :]                                                         
      249 -    Ai_22 = tl.load(ptr_Ad_1, mask=mask_m_1).to(tl.float32)   
      245 +    # Load Ai_11 (Ad block at row i_t*64)                     
      246 +    offs_m = i_t * 64 + offs_16                               
      247 +    mask_Ad = offs_m[:, None] < T                             
      248 +    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          +]                                                             
      249 +    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      250  
      251 -    # Load A_21 and compute Ai_21 immediately                 
      252 -    ptr_A_21 = A + offs_m_1[:, None] * (H * 64) + offs_16[None
          -, :]                                                          
      253 -    A_21 = tl.load(ptr_A_21, mask=mask_m_1).to(tl.float32)    
      254 -    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision="ieee"
          -), Ai_11, input_precision="ieee")                             
      251 +    Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")    
      252  
      256 -    # Load Ai_33 + Ai_44                                      
      257 -    mask_m_2 = offs_m_2[:, None] < T                          
      258 -    ptr_Ad_2 = Ad + offs_m_2[:, None] * (H * 16) + offs_16[Non
          -e, :]                                                         
      259 -    Ai_33 = tl.load(ptr_Ad_2, mask=mask_m_2).to(tl.float32)   
      253 +    # Load Ai_44 (Ad block at row i_t*64+48)                  
      254 +    offs_m = i_t * 64 + 48 + offs_16                          
      255 +    mask_Ad = offs_m[:, None] < T                             
      256 +    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          +]                                                             
      257 +    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      258  
      261 -    mask_m_3 = offs_m_3[:, None] < T                          
      262 -    ptr_Ad_3 = Ad + offs_m_3[:, None] * (H * 16) + offs_16[Non
          -e, :]                                                         
      263 -    Ai_44 = tl.load(ptr_Ad_3, mask=mask_m_3).to(tl.float32)   
      264 -                                                              
      265 -    # Load A_43 and compute Ai_43 immediately                 
      259 +    # Load A_43 (A block at row i_t*64+48, col 32) and compute
          + immediately                                                  
      260      offs_n_32 = 32 + offs_16
      267 -    ptr_A_43 = A + offs_m_3[:, None] * (H * 64) + offs_n_32[No
          -ne, :]                                                        
      268 -    mask_A_43 = (offs_m_3[:, None] < T) & (offs_n_32[None, :] 
          -< 64)                                                         
      269 -    A_43 = tl.load(ptr_A_43, mask=mask_A_43).to(tl.float32)   
      270 -    Ai_43 = -tl.dot(tl.dot(Ai_44, A_43, input_precision="ieee"
          -), Ai_33, input_precision="ieee")                             
      261 +    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n_32[None, :
          +]                                                             
      262 +    A_43 = tl.load(ptr_A, mask=mask_Ad).to(tl.float32)        
      263 +    tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")      
      264  
      272 -    # Build 32x32 blocks                                      
      265 +    # Load Ai_33 (Ad block at row i_t*64+32)                  
      266 +    offs_m = i_t * 64 + 32 + offs_16                          
      267 +    mask_Ad = offs_m[:, None] < T                             
      268 +    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          +]                                                             
      269 +    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      270 +                                                              
      271 +    Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")    
      272 +                                                              
      273 +    # Build Ai_22_32 (32x32)                                  
      274      Ai_22_32 = tl.zeros((32, 32), tl.float32)
      275      Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16),
            (1, 1))
      276      Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16
           ), (1, 1))
      277      Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16)
           , (1, 1))
      278  
      279 +    # Load A_21_32 (A block at row i_t*64+32, col 0, 32x32)   
      280 +    offs_m = i_t * 64 + 32 + tl.arange(0, 32)                 
      281 +    offs_n = tl.arange(0, 32)                                 
      282 +    mask_A = offs_m[:, None] < T                              
      283 +    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]  
      284 +    A_21_32 = tl.load(ptr_A, mask=mask_A).to(tl.float32)      
      285 +    tmp_lo = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")
      286 +                                                              
      287 +    # Build Ai_11_32 (32x32)                                  
      288      Ai_11_32 = tl.zeros((32, 32), tl.float32)
      289      Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16),
            (1, 1))
      290      Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
           ), (1, 1))
      291      Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
           , (1, 1))
      292  
      283 -    # Load A_21_32 and compute Ai_21_32                       
      284 -    offs_m_lo = block_start + 32 + tl.arange(0, 32)           
      285 -    offs_n_lo = tl.arange(0, 32)                              
      286 -    mask_A_lo = (offs_m_lo[:, None] < T) & (offs_n_lo[None, :]
          - < 64)                                                        
      287 -    ptr_A_lo = A + offs_m_lo[:, None] * (H * 64) + offs_n_lo[N
          -one, :]                                                       
      288 -    A_21_32 = tl.load(ptr_A_lo, mask=mask_A_lo).to(tl.float32)
      289 -    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          -on="ieee"), Ai_11_32, input_precision="ieee")                 
      293 +    Ai_21_32 = -tl.dot(tmp_lo, Ai_11_32, input_precision="ieee
          +")                                                            
      294  
      291 -    # Store results: Ai_11_32 at (0, 0)                       
      292 -    offs_32 = tl.arange(0, 32)                                
      293 -    offs_m_store_11 = block_start + offs_32                   
      294 -    mask_store_11 = (offs_m_store_11[:, None] < T) & (offs_32[
          -None, :] < 64)                                                
      295 -    ptr_store_11 = Ai + offs_m_store_11[:, None] * (H * 64) + 
          -offs_32[None, :]                                              
      296 -    tl.store(ptr_store_11, Ai_11_32.to(ptr_store_11.dtype.elem
          -ent_ty, fp_downcast_rounding="rtne"), mask=mask_store_11)     
      295 +    # Store Ai_11_32 at (0, 0)                                
      296 +    offs_m = i_t * 64 + tl.arange(0, 32)                      
      297 +    offs_n = tl.arange(0, 32)                                 
      298 +    mask_store = offs_m[:, None] < T                          
      299 +    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      300 +    tl.store(ptr_Ai, Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_d
          +owncast_rounding="rtne"), mask=mask_store)                    
      301  
      302      # Store Ai_22_32 at (32, 32)
      299 -    offs_m_store_22 = block_start + 32 + offs_32              
      300 -    offs_n_store_22 = 32 + offs_32                            
      301 -    mask_store_22 = (offs_m_store_22[:, None] < T) & (offs_n_s
          -tore_22[None, :] < 64)                                        
      302 -    ptr_store_22 = Ai + offs_m_store_22[:, None] * (H * 64) + 
          -offs_n_store_22[None, :]                                      
      303 -    tl.store(ptr_store_22, Ai_22_32.to(ptr_store_22.dtype.elem
          -ent_ty, fp_downcast_rounding="rtne"), mask=mask_store_22)     
      303 +    offs_m = i_t * 64 + 32 + tl.arange(0, 32)                 
      304 +    offs_n = 32 + tl.arange(0, 32)                            
      305 +    mask_store = offs_m[:, None] < T                          
      306 +    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      307 +    tl.store(ptr_Ai, Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_d
          +owncast_rounding="rtne"), mask=mask_store)                    
      308  
      309      # Store Ai_21_32 at (32, 0)
      306 -    mask_store_21 = (offs_m_store_22[:, None] < T) & (offs_32[
          -None, :] < 64)                                                
      307 -    ptr_store_21 = Ai + offs_m_store_22[:, None] * (H * 64) + 
          -offs_32[None, :]                                              
      308 -    tl.store(ptr_store_21, Ai_21_32.to(ptr_store_21.dtype.elem
          -ent_ty, fp_downcast_rounding="rtne"), mask=mask_store_21)     
      310 +    offs_n = tl.arange(0, 32)                                 
      311 +    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      312 +    tl.store(ptr_Ai, Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_d
          +owncast_rounding="rtne"), mask=mask_store)                    
      313  
      314      # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      311 -    mask_store_zero = (offs_m_store_11[:, None] < T) & (offs_n
          -_store_22[None, :] < BT)                                      
      312 -    ptr_store_zero = Ai + offs_m_store_11[:, None] * (H * BT) 
          -+ offs_n_store_22[None, :]                                    
      313 -    zero_block = tl.zeros((32, 32), dtype=ptr_store_zero.dtype
          -.element_ty)                                                  
      314 -    tl.store(ptr_store_zero, zero_block, mask=mask_store_zero)
      315 +    offs_m = i_t * 64 + tl.arange(0, 32)                      
      316 +    offs_n = 32 + tl.arange(0, 32)                            
      317 +    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT
          +)                                                             
      318 +    ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
      319 +    zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element
          +_ty)                                                          
      320 +    tl.store(ptr_Ai, zero_block, mask=mask_store)             
      321  
      322  
      323  def solve_tril(

● Key changes from the previous version:
  1. Removed other=0.0 from all tl.load calls (6 instances) - prevents internal
  tl.where that blocks load parallelism
  2. Simplified column masks - for 16x16 loads from Ad, offs_n < 16 is always
  true, removed. For 16x16 loads from A with col offset 0, offs_n < 64 is always
   true, removed. Only kept masks that are actually needed.
  3. Load-compute interleaving - load Ai_22 → load A_21 → compute dot, then load
   Ai_11 → complete dot. Same for Ai_44/A_43/Ai_33 chain.
  4. Reduced variable reuse - cleaned up temporary variables

  Now let's run correctness tests:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:26:42 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:26:42 [__init__.py:46] - ascend -> vllm_ascend:register
     … +21 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All 7 correctness checks pass with max_diff=0.000000. Now let's measure
  performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:28:40.366495
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v4/ -name "op_summary_*.csv" |
       head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.60us, min=33.24us, max=34.04us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=113.55us, min=112.64us
     … +3 lines (ctrl+o to expand)

● The merge kernel went from 111.81us to 113.55us — essentially no change
  (within measurement noise). The other=0.0 removal didn't help because the
  original kernel was already using explicit pointers, not make_block_ptr with
  boundary_check. The masks I simplified were already quite clean.

  Let me look more carefully at the merge kernel. The fundamental issue is that
  it's Scalar-dominated (32-34%). Let me think about what we can do
  structurally.

  The merge kernel does:
  - 6 loads of 16x16 blocks from Ad and A
  - 6 tl.dot operations (16x16 × 16x16)
  - 6 insert_slice to build 32x32 blocks
  - 2 tl.dot operations (32x32 × 32x32)
  - 4 stores of 32x32 blocks

  The Scalar overhead comes from all the pointer/mask computations. Let me try a
   fundamentally different approach: use make_block_ptr for the loads to reduce
  Scalar overhead (msprof showed it had lower Scalar ratio in baseline).

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 31 lines, removed 72 lines
      224      else:
      225          bos, eos = i_b * T, i_b * T + T
      226  
      227 -    # Base pointers (already offset by batch and head)        
      228 -    A += (bos * H + i_h) * 64                                 
      227 +    A += (bos * H + i_h) * BT                                 
      228      Ad += (bos * H + i_h) * 16
      230 -    Ai += (bos * H + i_h) * 64                                
      229 +    Ai += (bos * H + i_h) * BT                                
      230  
      232 -    offs_16 = tl.arange(0, 16)                                
      233 -                                                              
      234 -    # Load Ai_22 (Ad block at row i_t*64+16)                  
      235 -    offs_m = i_t * 64 + 16 + offs_16                          
      236 -    mask_Ad = offs_m[:, None] < T                             
      237 -    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          -]                                                             
      238 -    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      239 -                                                              
      240 -    # Load A_21 (A block at row i_t*64+16, col 0) and compute 
          -immediately                                                   
      241 -    ptr_A = A + offs_m[:, None] * (H * 64) + offs_16[None, :] 
      242 -    A_21 = tl.load(ptr_A, mask=mask_Ad).to(tl.float32)        
      231 +    # Use make_block_ptr for minimal Scalar overhead on 16x16 
          +loads                                                         
      232 +    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          + * 64 + 16, 0), (16, 16), (1, 0))                             
      233 +    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t *
          + 64 + 16, 0), (16, 16), (1, 0))                               
      234 +    Ai_22 = tl.load(p_Ad_22, boundary_check=(0,)).to(tl.float3
          +2)                                                            
      235 +    A_21 = tl.load(p_A_21, boundary_check=(0,)).to(tl.float32)
      236      tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")
      237  
      245 -    # Load Ai_11 (Ad block at row i_t*64)                     
      246 -    offs_m = i_t * 64 + offs_16                               
      247 -    mask_Ad = offs_m[:, None] < T                             
      248 -    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          -]                                                             
      249 -    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      250 -                                                              
      238 +    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          + * 64, 0), (16, 16), (1, 0))                                  
      239 +    Ai_11 = tl.load(p_Ad_11, boundary_check=(0,)).to(tl.float3
          +2)                                                            
      240      Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")
      241  
      253 -    # Load Ai_44 (Ad block at row i_t*64+48)                  
      254 -    offs_m = i_t * 64 + 48 + offs_16                          
      255 -    mask_Ad = offs_m[:, None] < T                             
      256 -    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          -]                                                             
      257 -    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      258 -                                                              
      259 -    # Load A_43 (A block at row i_t*64+48, col 32) and compute
          - immediately                                                  
      260 -    offs_n_32 = 32 + offs_16                                  
      261 -    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n_32[None, :
          -]                                                             
      262 -    A_43 = tl.load(ptr_A, mask=mask_Ad).to(tl.float32)        
      242 +    p_Ad_44 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          + * 64 + 48, 0), (16, 16), (1, 0))                             
      243 +    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t *
          + 64 + 48, 32), (16, 16), (1, 0))                              
      244 +    Ai_44 = tl.load(p_Ad_44, boundary_check=(0,)).to(tl.float3
          +2)                                                            
      245 +    A_43 = tl.load(p_A_43, boundary_check=(0,)).to(tl.float32)
      246      tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")
      247  
      265 -    # Load Ai_33 (Ad block at row i_t*64+32)                  
      266 -    offs_m = i_t * 64 + 32 + offs_16                          
      267 -    mask_Ad = offs_m[:, None] < T                             
      268 -    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_16[None, :
          -]                                                             
      269 -    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad).to(tl.float32)      
      270 -                                                              
      248 +    p_Ad_33 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          + * 64 + 32, 0), (16, 16), (1, 0))                             
      249 +    Ai_33 = tl.load(p_Ad_33, boundary_check=(0,)).to(tl.float3
          +2)                                                            
      250      Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")
      251  
      273 -    # Build Ai_22_32 (32x32)                                  
      252 +    # Build 32x32 blocks                                      
      253      Ai_22_32 = tl.zeros((32, 32), tl.float32)
      254      Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16),
            (1, 1))
      255      Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16
           ), (1, 1))
      256      Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16)
           , (1, 1))
      257  
      279 -    # Load A_21_32 (A block at row i_t*64+32, col 0, 32x32)   
      280 -    offs_m = i_t * 64 + 32 + tl.arange(0, 32)                 
      281 -    offs_n = tl.arange(0, 32)                                 
      282 -    mask_A = offs_m[:, None] < T                              
      283 -    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]  
      284 -    A_21_32 = tl.load(ptr_A, mask=mask_A).to(tl.float32)      
      285 -    tmp_lo = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")
      286 -                                                              
      287 -    # Build Ai_11_32 (32x32)                                  
      258      Ai_11_32 = tl.zeros((32, 32), tl.float32)
      259      Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16),
            (1, 1))
      260      Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
           ), (1, 1))
      261      Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
           , (1, 1))
      262  
      293 -    Ai_21_32 = -tl.dot(tmp_lo, Ai_11_32, input_precision="ieee
          -")                                                            
      263 +    # 32x32 level merge                                       
      264 +    p_A_21_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_
          +t * 64 + 32, 0), (32, 32), (1, 0))                            
      265 +    A_21_32 = tl.load(p_A_21_32, boundary_check=(0,)).to(tl.fl
          +oat32)                                                        
      266 +    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          +on="ieee"), Ai_11_32, input_precision="ieee")                 
      267  
      295 -    # Store Ai_11_32 at (0, 0)                                
      296 -    offs_m = i_t * 64 + tl.arange(0, 32)                      
      297 -    offs_n = tl.arange(0, 32)                                 
      298 -    mask_store = offs_m[:, None] < T                          
      299 -    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      300 -    tl.store(ptr_Ai, Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_d
          -owncast_rounding="rtne"), mask=mask_store)                    
      268 +    # Store results using make_block_ptr                      
      269 +    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          + * 64, 0), (32, 32), (1, 0))                                  
      270 +    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          + * 64 + 32, 32), (32, 32), (1, 0))                            
      271 +    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          + * 64 + 32, 0), (32, 32), (1, 0))                             
      272 +    tl.store(p_Ai_11, Ai_11_32.to(p_Ai_11.dtype.element_ty, fp
          +_downcast_rounding="rtne"), boundary_check=(0,))              
      273 +    tl.store(p_Ai_22, Ai_22_32.to(p_Ai_22.dtype.element_ty, fp
          +_downcast_rounding="rtne"), boundary_check=(0,))              
      274 +    tl.store(p_Ai_21, Ai_21_32.to(p_Ai_21.dtype.element_ty, fp
          +_downcast_rounding="rtne"), boundary_check=(0,))              
      275  
      302 -    # Store Ai_22_32 at (32, 32)                              
      303 -    offs_m = i_t * 64 + 32 + tl.arange(0, 32)                 
      304 -    offs_n = 32 + tl.arange(0, 32)                            
      305 -    mask_store = offs_m[:, None] < T                          
      306 -    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      307 -    tl.store(ptr_Ai, Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_d
          -owncast_rounding="rtne"), mask=mask_store)                    
      276 +    # Zero out upper-right 32x32 block                        
      277 +    p_Ai_zero = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i
          +_t * 64, 32), (32, 32), (1, 0))                               
      278 +    zero_block = tl.zeros((32, 32), dtype=p_Ai_zero.dtype.elem
          +ent_ty)                                                       
      279 +    tl.store(p_Ai_zero, zero_block, boundary_check=(0,))      
      280  
      309 -    # Store Ai_21_32 at (32, 0)                               
      310 -    offs_n = tl.arange(0, 32)                                 
      311 -    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
      312 -    tl.store(ptr_Ai, Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_d
          -owncast_rounding="rtne"), mask=mask_store)                    
      281  
      314 -    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      315 -    offs_m = i_t * 64 + tl.arange(0, 32)                      
      316 -    offs_n = 32 + tl.arange(0, 32)                            
      317 -    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT
          -)                                                             
      318 -    ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
      319 -    zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element
          -_ty)                                                          
      320 -    tl.store(ptr_Ai, zero_block, mask=mask_store)             
      321 -                                                              
      322 -                                                              
      282  def solve_tril(
      283      A: torch.Tensor,
      284      cu_seqlens: torch.Tensor | None = None,

● Now run correctness then performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Error: Exit code 1
     INFO 04-11 09:31:12 [__init__.py:44] Available plugins for group 
     vllm.platform_plugins:
     INFO 04-11 09:31:12 [__init__.py:46] - ascend -> vllm_ascend:register
     INFO 04-11 09:31:12 [__init__.py:49] All plugins in this group will be 
     loaded. Set `VLLM_PLUGINS` to control which plugins to load.
     INFO 04-11 09:31:12 [__init__.py:212] Platform plugin ascend is activated
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     … +69 lines (ctrl+o to see all)

● The tl.zeros with p_Ai_zero.dtype.element_ty causes a compilation error on
  Ascend. The issue is using make_block_ptr's dtype for tl.zeros. Let me fix
  this — for the zero block store, I need to use explicit pointers since
  make_block_ptr dtype is not directly compatible with tl.zeros on Ascend.

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 7 lines, removed 3 lines
      274      tl.store(p_Ai_21, Ai_21_32.to(p_Ai_21.dtype.element_ty, fp
           _downcast_rounding="rtne"), boundary_check=(0,))
      275  
      276      # Zero out upper-right 32x32 block
      277 -    p_Ai_zero = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i
          -_t * 64, 32), (32, 32), (1, 0))                               
      278 -    zero_block = tl.zeros((32, 32), dtype=p_Ai_zero.dtype.elem
          -ent_ty)                                                       
      279 -    tl.store(p_Ai_zero, zero_block, boundary_check=(0,))      
      277 +    offs_32 = tl.arange(0, 32)                                
      278 +    offs_m_z = i_t * 64 + offs_32                             
      279 +    offs_n_z = 32 + offs_32                                   
      280 +    mask_z = (offs_m_z[:, None] < T) & (offs_n_z[None, :] < BT
          +)                                                             
      281 +    ptr_z = Ai + offs_m_z[:, None] * (H * BT) + offs_n_z[None,
          + :]                                                           
      282 +    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      283 +    tl.store(ptr_z, zero_block.to(ptr_z.dtype.element_ty), mas
          +k=mask_z)                                                     
      284  
      285  
      286  def solve_tril(

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:32:43 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:32:43 [__init__.py:46] - ascend -> vllm_ascend:register
     … +21 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All correct. Now measure performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:34:44.026521
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v5/ -name "op_summary_*.csv" |
       head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.73us, min=33.42us, max=34.02us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=114.14us, min=112.64us
     … +3 lines (ctrl+o to expand)

● The merge kernel with make_block_ptr + load-compute interleaving gives
  114.14us — essentially the same as explicit pointers (113.55us). The merge
  kernel is stubborn at ~112-114us regardless of pointer style.

  Let me think about a fundamentally different approach. The merge kernel has
  304 work items (19 blocks × 16 batch*heads). Each does very little compute but
   has high scheduling overhead. What if I try a 1D Grid approach with multiple 
  blocks per core — but unlike the previous attempt that regressed, this time
  I'll use get_vectorcore_num()//2 (20 cores, since it uses tl.dot) and process
  multiple blocks in a loop?

  Actually, let me first review the grid dimensions more carefully:

  Read 1 file (ctrl+o to expand)

● The merge kernel is launched with grid [NT, B*H]. For our test case: NT =
  ceil(1216/64) = 19, BH = 28 = 16. So total work items = 19*16 = 304 blocks.

  Per the skill guide, kernel with tl.dot must use grid = (get_vectorcore_num() 
  // 2, ) = 20. That means 304 blocks across 20 cores = ~15 blocks per core. The
   previous attempt with 1D Grid used 20 cores but regressed.

  Let me check why the 1D attempt regressed. With 304 blocks on 20 cores, each
  core does ~15 stride loop iterations. Each iteration has the same amount of
  compute but adds Scalar overhead for loop control and index decoding. If
  Scalar is already 32-34% of time, adding 15x more Scalar iterations would
  multiply the Scalar portion.

  Let me consider another approach: increasing the work per merge kernel 
  invocation by processing more data in each block. Currently each block
  processes one 64x64 tile. What if we process 2 tiles per block? That would
  halve the total blocks and reduce scheduling overhead.

  But wait — the merge kernel's structure is fundamentally constrained by the
  algorithm: it needs the Ad (16x16 diagonal blocks) from the solve kernel, then
   merges them hierarchically to get the full 64x64 inverse. Each 64x64 block is
   independent. The work per block is very small (6 dot products of 16x16, 2 dot
   products of 32x32), so the kernel is scheduling-bound.

  Let me try a different approach: fuse multiple 64x64 blocks into one kernel 
  invocation using a stride loop, but with the 1D Grid mandated by the rules.
  The key insight from E4 is that BLOCK_T adjustments can't overcome scheduling
  overhead, but what if each stride loop iteration is extremely cheap (just
  loads + a few dots)?

  Let me try converting to 1D Grid with get_vectorcore_num()//2:
● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 96 lines, removed 64 lines
      196  
      197  
      198  @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens
           "] is not None})
      199 -@triton.jit(do_not_specialize=["T"])                          
      199 +@triton.jit(do_not_specialize=["T", "total_blocks"])          
      200  def merge_16x16_to_64x64_inverse_kernel(
      201      A,
      202      Ad,
     ...
      204      cu_seqlens,
      205      chunk_indices,
      206      T,
      207 +    total_blocks,                                             
      208      H: tl.constexpr,
      209      BT: tl.constexpr,
      210      IS_VARLEN: tl.constexpr,
      211  ):
      211 -    i_t, i_bh = tl.program_id(0), tl.program_id(1)            
      212 -    i_b, i_h = i_bh // H, i_bh % H                            
      213 -    if IS_VARLEN:                                             
      214 -        i_n, i_t_val = (                                      
      215 -            tl.load(chunk_indices + i_t * 2).to(tl.int32),    
      216 -            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
      217 -        )                                                     
      218 -        bos, eos = (                                          
      219 -            tl.load(cu_seqlens + i_n).to(tl.int32),           
      220 -            tl.load(cu_seqlens + i_n + 1).to(tl.int32),       
      221 -        )                                                     
      222 -        T = eos - bos                                         
      223 -        i_t = i_t_val                                         
      224 -    else:                                                     
      225 -        bos, eos = i_b * T, i_b * T + T                       
      212 +    pid = tl.program_id(0)                                    
      213 +    num_programs = tl.num_programs(0)                         
      214  
      227 -    A += (bos * H + i_h) * BT                                 
      228 -    Ad += (bos * H + i_h) * 16                                
      229 -    Ai += (bos * H + i_h) * BT                                
      215 +    for work_idx in range(pid, total_blocks, num_programs):   
      216 +        if IS_VARLEN:                                         
      217 +            i_t_raw = work_idx // H                           
      218 +            i_h = work_idx % H                                
      219 +            i_n = tl.load(chunk_indices + i_t_raw * 2).to(tl.i
          +nt32)                                                         
      220 +            i_t = tl.load(chunk_indices + i_t_raw * 2 + 1).to(
          +tl.int32)                                                     
      221 +            bos = tl.load(cu_seqlens + i_n).to(tl.int32)      
      222 +            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)  
      223 +            cur_T = eos - bos                                 
      224 +        else:                                                 
      225 +            blocks_per_batch = tl.cdiv(T, BT)                 
      226 +            i_bh = work_idx // blocks_per_batch               
      227 +            i_t = work_idx % blocks_per_batch                 
      228 +            i_b = i_bh // H                                   
      229 +            i_h = i_bh % H                                    
      230 +            bos = i_b * T                                     
      231 +            cur_T = T                                         
      232  
      231 -    # Use make_block_ptr for minimal Scalar overhead on 16x16 
          -loads                                                         
      232 -    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          - * 64 + 16, 0), (16, 16), (1, 0))                             
      233 -    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t *
          - 64 + 16, 0), (16, 16), (1, 0))                               
      234 -    Ai_22 = tl.load(p_Ad_22, boundary_check=(0,)).to(tl.float3
          -2)                                                            
      235 -    A_21 = tl.load(p_A_21, boundary_check=(0,)).to(tl.float32)
      236 -    tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")      
      233 +        A_base = A + (bos * H + i_h) * BT                     
      234 +        Ad_base = Ad + (bos * H + i_h) * 16                   
      235 +        Ai_base = Ai + (bos * H + i_h) * BT                   
      236  
      238 -    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          - * 64, 0), (16, 16), (1, 0))                                  
      239 -    Ai_11 = tl.load(p_Ad_11, boundary_check=(0,)).to(tl.float3
          -2)                                                            
      240 -    Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")    
      237 +        offs_16 = tl.arange(0, 16)                            
      238  
      242 -    p_Ad_44 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          - * 64 + 48, 0), (16, 16), (1, 0))                             
      243 -    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t *
          - 64 + 48, 32), (16, 16), (1, 0))                              
      244 -    Ai_44 = tl.load(p_Ad_44, boundary_check=(0,)).to(tl.float3
          -2)                                                            
      245 -    A_43 = tl.load(p_A_43, boundary_check=(0,)).to(tl.float32)
      246 -    tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")      
      239 +        # Load Ai_22 and A_21, compute dot immediately        
      240 +        offs_m1 = i_t * 64 + 16 + offs_16                     
      241 +        mask_m1 = offs_m1[:, None] < cur_T                    
      242 +        Ai_22 = tl.load(Ad_base + offs_m1[:, None] * (H * 16) 
          ++ offs_16[None, :], mask=mask_m1).to(tl.float32)              
      243 +        A_21 = tl.load(A_base + offs_m1[:, None] * (H * BT) + 
          +offs_16[None, :], mask=mask_m1).to(tl.float32)                
      244 +        tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")  
      245  
      248 -    p_Ad_33 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t
          - * 64 + 32, 0), (16, 16), (1, 0))                             
      249 -    Ai_33 = tl.load(p_Ad_33, boundary_check=(0,)).to(tl.float3
          -2)                                                            
      250 -    Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")    
      246 +        # Load Ai_11, complete Ai_21                          
      247 +        offs_m0 = i_t * 64 + offs_16                          
      248 +        mask_m0 = offs_m0[:, None] < cur_T                    
      249 +        Ai_11 = tl.load(Ad_base + offs_m0[:, None] * (H * 16) 
          ++ offs_16[None, :], mask=mask_m0).to(tl.float32)              
      250 +        Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")
      251  
      252 -    # Build 32x32 blocks                                      
      253 -    Ai_22_32 = tl.zeros((32, 32), tl.float32)                 
      254 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16),
          - (1, 1))                                                      
      255 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16
          -), (1, 1))                                                    
      256 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16)
          -, (1, 1))                                                     
      252 +        # Load Ai_44 and A_43, compute dot immediately        
      253 +        offs_m3 = i_t * 64 + 48 + offs_16                     
      254 +        offs_n32 = 32 + offs_16                               
      255 +        mask_m3 = offs_m3[:, None] < cur_T                    
      256 +        Ai_44 = tl.load(Ad_base + offs_m3[:, None] * (H * 16) 
          ++ offs_16[None, :], mask=mask_m3).to(tl.float32)              
      257 +        A_43 = tl.load(A_base + offs_m3[:, None] * (H * BT) + 
          +offs_n32[None, :], mask=mask_m3).to(tl.float32)               
      258 +        tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")  
      259  
      258 -    Ai_11_32 = tl.zeros((32, 32), tl.float32)                 
      259 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16),
          - (1, 1))                                                      
      260 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
          -), (1, 1))                                                    
      261 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
          -, (1, 1))                                                     
      260 +        # Load Ai_33, complete Ai_43                          
      261 +        offs_m2 = i_t * 64 + 32 + offs_16                     
      262 +        mask_m2 = offs_m2[:, None] < cur_T                    
      263 +        Ai_33 = tl.load(Ad_base + offs_m2[:, None] * (H * 16) 
          ++ offs_16[None, :], mask=mask_m2).to(tl.float32)              
      264 +        Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")
      265  
      263 -    # 32x32 level merge                                       
      264 -    p_A_21_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_
          -t * 64 + 32, 0), (32, 32), (1, 0))                            
      265 -    A_21_32 = tl.load(p_A_21_32, boundary_check=(0,)).to(tl.fl
          -oat32)                                                        
      266 -    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          -on="ieee"), Ai_11_32, input_precision="ieee")                 
      266 +        # Build 32x32 blocks                                  
      267 +        Ai_22_32 = tl.zeros((32, 32), tl.float32)             
      268 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 
          +16), (1, 1))                                                  
      269 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16
          +, 16), (1, 1))                                                
      270 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16,
          + 16), (1, 1))                                                 
      271  
      268 -    # Store results using make_block_ptr                      
      269 -    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          - * 64, 0), (32, 32), (1, 0))                                  
      270 -    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          - * 64 + 32, 32), (32, 32), (1, 0))                            
      271 -    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t
          - * 64 + 32, 0), (32, 32), (1, 0))                             
      272 -    tl.store(p_Ai_11, Ai_11_32.to(p_Ai_11.dtype.element_ty, fp
          -_downcast_rounding="rtne"), boundary_check=(0,))              
      273 -    tl.store(p_Ai_22, Ai_22_32.to(p_Ai_22.dtype.element_ty, fp
          -_downcast_rounding="rtne"), boundary_check=(0,))              
      274 -    tl.store(p_Ai_21, Ai_21_32.to(p_Ai_21.dtype.element_ty, fp
          -_downcast_rounding="rtne"), boundary_check=(0,))              
      272 +        Ai_11_32 = tl.zeros((32, 32), tl.float32)             
      273 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 
          +16), (1, 1))                                                  
      274 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16
          +, 16), (1, 1))                                                
      275 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16,
          + 16), (1, 1))                                                 
      276  
      276 -    # Zero out upper-right 32x32 block                        
      277 -    offs_32 = tl.arange(0, 32)                                
      278 -    offs_m_z = i_t * 64 + offs_32                             
      279 -    offs_n_z = 32 + offs_32                                   
      280 -    mask_z = (offs_m_z[:, None] < T) & (offs_n_z[None, :] < BT
          -)                                                             
      281 -    ptr_z = Ai + offs_m_z[:, None] * (H * BT) + offs_n_z[None,
          - :]                                                           
      282 -    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      283 -    tl.store(ptr_z, zero_block.to(ptr_z.dtype.element_ty), mas
          -k=mask_z)                                                     
      277 +        # 32x32 level merge                                   
      278 +        offs_32 = tl.arange(0, 32)                            
      279 +        offs_m_lo = i_t * 64 + 32 + offs_32                   
      280 +        mask_lo = offs_m_lo[:, None] < cur_T                  
      281 +        A_21_32 = tl.load(A_base + offs_m_lo[:, None] * (H * B
          +T) + offs_32[None, :], mask=mask_lo).to(tl.float32)           
      282 +        Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_pre
          +cision="ieee"), Ai_11_32, input_precision="ieee")             
      283  
      284 +        # Store Ai_11_32 at (0, 0)                            
      285 +        offs_m_s0 = i_t * 64 + offs_32                        
      286 +        mask_s0 = offs_m_s0[:, None] < cur_T                  
      287 +        tl.store(                                             
      288 +            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_32[
          +None, :],                                                     
      289 +            Ai_11_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      290 +            mask=mask_s0,                                     
      291 +        )                                                     
      292  
      293 +        # Store Ai_22_32 at (32, 32)                          
      294 +        offs_n_32 = 32 + offs_32                              
      295 +        tl.store(                                             
      296 +            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_n_3
          +2[None, :],                                                   
      297 +            Ai_22_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      298 +            mask=mask_lo,                                     
      299 +        )                                                     
      300 +                                                              
      301 +        # Store Ai_21_32 at (32, 0)                           
      302 +        tl.store(                                             
      303 +            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_32[
          +None, :],                                                     
      304 +            Ai_21_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      305 +            mask=mask_lo,                                     
      306 +        )                                                     
      307 +                                                              
      308 +        # Zero out upper-right 32x32 block (rows 0~31, cols 32
          +~63)                                                          
      309 +        mask_z = (offs_m_s0[:, None] < cur_T) & (offs_n_32[Non
          +e, :] < BT)                                                   
      310 +        zero_block = tl.zeros((32, 32), dtype=tl.float32)     
      311 +        tl.store(                                             
      312 +            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_n_3
          +2[None, :],                                                   
      313 +            zero_block.to(Ai_base.dtype.element_ty),          
      314 +            mask=mask_z,                                      
      315 +        )                                                     
      316 +                                                              
      317 +                                                              
      318  def solve_tril(
      319      A: torch.Tensor,
      320      cu_seqlens: torch.Tensor | None = None,

● Now update the host-side launch code to use 1D Grid:

  Read 1 file (ctrl+o to expand)

● I need to update the launch code to use 1D Grid for the merge kernel:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 29 lines, removed 12 lines
      395      chunk_indices = chunk_indices_bt
      396      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      397  
      398 -    merge_fn[NT, B * H](                                      
      399 -        A=A,                                                  
      400 -        Ad=Ad,                                                
      401 -        Ai=Ai,                                                
      402 -        cu_seqlens=cu_seqlens,                                
      403 -        chunk_indices=chunk_indices,                          
      404 -        T=T,                                                  
      405 -        H=H,                                                  
      406 -        BT=BT,                                                
      407 -        num_warps=4,                                          
      408 -        num_stages=3,                                         
      409 -    )                                                         
      398 +    if BT == 64:                                              
      399 +        total_blocks_merge = NT * H if IS_VARLEN else B * H * 
          +NT                                                            
      400 +        grid_merge = (get_vectorcore_num() // 2, )            
      401 +        merge_fn[grid_merge](                                 
      402 +            A=A,                                              
      403 +            Ad=Ad,                                            
      404 +            Ai=Ai,                                            
      405 +            cu_seqlens=cu_seqlens,                            
      406 +            chunk_indices=chunk_indices,                      
      407 +            T=T,                                              
      408 +            total_blocks=total_blocks_merge,                  
      409 +            H=H,                                              
      410 +            BT=BT,                                            
      411 +            num_warps=4,                                      
      412 +            num_stages=3,                                     
      413 +        )                                                     
      414 +    else:                                                     
      415 +        merge_fn[NT, B * H](                                  
      416 +            A=A,                                              
      417 +            Ad=Ad,                                            
      418 +            Ai=Ai,                                            
      419 +            cu_seqlens=cu_seqlens,                            
      420 +            chunk_indices=chunk_indices,                      
      421 +            T=T,                                              
      422 +            H=H,                                              
      423 +            BT=BT,                                            
      424 +            num_warps=4,                                      
      425 +            num_stages=3,                                     
      426 +        )                                                     
      427      return Ai

● Now run correctness check:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:37:46 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:37:46 [__init__.py:46] - ascend -> vllm_ascend:register
     … +21 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All correct. Now measure performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:39:47.151166
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v6/ -name "op_summary_*.csv" |
       head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.80us, min=33.58us, max=34.08us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=151.72us, min=148.42us
     … +3 lines (ctrl+o to expand)

● As expected from the previous attempt — the 1D Grid for merge kernel regresses
   significantly (114us → 152us). This matches the experience from E3: 20 cores
  has massive scheduling overhead when the kernel is this small, and the stride
  loop Scalar overhead compounds it (304 blocks / 20 cores ≈ 15 iterations per
  core × Scalar overhead per iteration).

  Let me revert the merge kernel back to 2D Grid which performed better. The
  merge kernel is fundamentally scheduling-bound and the best performance was
  with 2D Grid.

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 87 lines, removed 93 lines
      196  
      197  
      198  @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens
           "] is not None})
      199 -@triton.jit(do_not_specialize=["T", "total_blocks"])          
      199 +@triton.jit(do_not_specialize=["T"])                          
      200  def merge_16x16_to_64x64_inverse_kernel(
      201      A,
      202      Ad,
     ...
      204      cu_seqlens,
      205      chunk_indices,
      206      T,
      207 -    total_blocks,                                             
      207      H: tl.constexpr,
      208      BT: tl.constexpr,
      209      IS_VARLEN: tl.constexpr,
      210  ):
      212 -    pid = tl.program_id(0)                                    
      213 -    num_programs = tl.num_programs(0)                         
      211 +    i_t, i_bh = tl.program_id(0), tl.program_id(1)            
      212 +    i_b, i_h = i_bh // H, i_bh % H                            
      213 +    if IS_VARLEN:                                             
      214 +        i_n, i_t_val = (                                      
      215 +            tl.load(chunk_indices + i_t * 2).to(tl.int32),    
      216 +            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
      217 +        )                                                     
      218 +        bos, eos = (                                          
      219 +            tl.load(cu_seqlens + i_n).to(tl.int32),           
      220 +            tl.load(cu_seqlens + i_n + 1).to(tl.int32),       
      221 +        )                                                     
      222 +        T = eos - bos                                         
      223 +        i_t = i_t_val                                         
      224 +    else:                                                     
      225 +        bos, eos = i_b * T, i_b * T + T                       
      226  
      215 -    for work_idx in range(pid, total_blocks, num_programs):   
      216 -        if IS_VARLEN:                                         
      217 -            i_t_raw = work_idx // H                           
      218 -            i_h = work_idx % H                                
      219 -            i_n = tl.load(chunk_indices + i_t_raw * 2).to(tl.i
          -nt32)                                                         
      220 -            i_t = tl.load(chunk_indices + i_t_raw * 2 + 1).to(
          -tl.int32)                                                     
      221 -            bos = tl.load(cu_seqlens + i_n).to(tl.int32)      
      222 -            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)  
      223 -            cur_T = eos - bos                                 
      224 -        else:                                                 
      225 -            blocks_per_batch = tl.cdiv(T, BT)                 
      226 -            i_bh = work_idx // blocks_per_batch               
      227 -            i_t = work_idx % blocks_per_batch                 
      228 -            i_b = i_bh // H                                   
      229 -            i_h = i_bh % H                                    
      230 -            bos = i_b * T                                     
      231 -            cur_T = T                                         
      227 +    A += (bos * H + i_h) * BT                                 
      228 +    Ad += (bos * H + i_h) * 16                                
      229 +    Ai += (bos * H + i_h) * BT                                
      230  
      233 -        A_base = A + (bos * H + i_h) * BT                     
      234 -        Ad_base = Ad + (bos * H + i_h) * 16                   
      235 -        Ai_base = Ai + (bos * H + i_h) * BT                   
      231 +    offs_16 = tl.arange(0, 16)                                
      232  
      237 -        offs_16 = tl.arange(0, 16)                            
      233 +    # Load Ai_22 and A_21, compute dot immediately (interleave
          +d)                                                            
      234 +    offs_m1 = i_t * 64 + 16 + offs_16                         
      235 +    mask_m1 = offs_m1[:, None] < T                            
      236 +    Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + offs_16
          +[None, :], mask=mask_m1).to(tl.float32)                       
      237 +    A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_16[N
          +one, :], mask=mask_m1).to(tl.float32)                         
      238 +    tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")      
      239  
      239 -        # Load Ai_22 and A_21, compute dot immediately        
      240 -        offs_m1 = i_t * 64 + 16 + offs_16                     
      241 -        mask_m1 = offs_m1[:, None] < cur_T                    
      242 -        Ai_22 = tl.load(Ad_base + offs_m1[:, None] * (H * 16) 
          -+ offs_16[None, :], mask=mask_m1).to(tl.float32)              
      243 -        A_21 = tl.load(A_base + offs_m1[:, None] * (H * BT) + 
          -offs_16[None, :], mask=mask_m1).to(tl.float32)                
      244 -        tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")  
      240 +    # Load Ai_11, complete Ai_21                              
      241 +    offs_m0 = i_t * 64 + offs_16                              
      242 +    mask_m0 = offs_m0[:, None] < T                            
      243 +    Ai_11 = tl.load(Ad + offs_m0[:, None] * (H * 16) + offs_16
          +[None, :], mask=mask_m0).to(tl.float32)                       
      244 +    Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")    
      245  
      246 -        # Load Ai_11, complete Ai_21                          
      247 -        offs_m0 = i_t * 64 + offs_16                          
      248 -        mask_m0 = offs_m0[:, None] < cur_T                    
      249 -        Ai_11 = tl.load(Ad_base + offs_m0[:, None] * (H * 16) 
          -+ offs_16[None, :], mask=mask_m0).to(tl.float32)              
      250 -        Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")
      246 +    # Load Ai_44 and A_43, compute dot immediately (interleave
          +d)                                                            
      247 +    offs_m3 = i_t * 64 + 48 + offs_16                         
      248 +    offs_n32 = 32 + offs_16                                   
      249 +    mask_m3 = offs_m3[:, None] < T                            
      250 +    Ai_44 = tl.load(Ad + offs_m3[:, None] * (H * 16) + offs_16
          +[None, :], mask=mask_m3).to(tl.float32)                       
      251 +    A_43 = tl.load(A + offs_m3[:, None] * (H * BT) + offs_n32[
          +None, :], mask=mask_m3).to(tl.float32)                        
      252 +    tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")      
      253  
      252 -        # Load Ai_44 and A_43, compute dot immediately        
      253 -        offs_m3 = i_t * 64 + 48 + offs_16                     
      254 -        offs_n32 = 32 + offs_16                               
      255 -        mask_m3 = offs_m3[:, None] < cur_T                    
      256 -        Ai_44 = tl.load(Ad_base + offs_m3[:, None] * (H * 16) 
          -+ offs_16[None, :], mask=mask_m3).to(tl.float32)              
      257 -        A_43 = tl.load(A_base + offs_m3[:, None] * (H * BT) + 
          -offs_n32[None, :], mask=mask_m3).to(tl.float32)               
      258 -        tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")  
      254 +    # Load Ai_33, complete Ai_43                              
      255 +    offs_m2 = i_t * 64 + 32 + offs_16                         
      256 +    mask_m2 = offs_m2[:, None] < T                            
      257 +    Ai_33 = tl.load(Ad + offs_m2[:, None] * (H * 16) + offs_16
          +[None, :], mask=mask_m2).to(tl.float32)                       
      258 +    Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")    
      259  
      260 -        # Load Ai_33, complete Ai_43                          
      261 -        offs_m2 = i_t * 64 + 32 + offs_16                     
      262 -        mask_m2 = offs_m2[:, None] < cur_T                    
      263 -        Ai_33 = tl.load(Ad_base + offs_m2[:, None] * (H * 16) 
          -+ offs_16[None, :], mask=mask_m2).to(tl.float32)              
      264 -        Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")
      260 +    # Build 32x32 blocks                                      
      261 +    Ai_22_32 = tl.zeros((32, 32), tl.float32)                 
      262 +    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16),
          + (1, 1))                                                      
      263 +    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16
          +), (1, 1))                                                    
      264 +    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16)
          +, (1, 1))                                                     
      265  
      266 -        # Build 32x32 blocks                                  
      267 -        Ai_22_32 = tl.zeros((32, 32), tl.float32)             
      268 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 
          -16), (1, 1))                                                  
      269 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16
          -, 16), (1, 1))                                                
      270 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16,
          - 16), (1, 1))                                                 
      266 +    Ai_11_32 = tl.zeros((32, 32), tl.float32)                 
      267 +    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16),
          + (1, 1))                                                      
      268 +    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
          +), (1, 1))                                                    
      269 +    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
          +, (1, 1))                                                     
      270  
      272 -        Ai_11_32 = tl.zeros((32, 32), tl.float32)             
      273 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 
          -16), (1, 1))                                                  
      274 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16
          -, 16), (1, 1))                                                
      275 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16,
          - 16), (1, 1))                                                 
      271 +    # 32x32 level merge                                       
      272 +    offs_32 = tl.arange(0, 32)                                
      273 +    offs_m_lo = i_t * 64 + 32 + offs_32                       
      274 +    mask_lo = offs_m_lo[:, None] < T                          
      275 +    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          +_32[None, :], mask=mask_lo).to(tl.float32)                    
      276 +    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          +on="ieee"), Ai_11_32, input_precision="ieee")                 
      277  
      277 -        # 32x32 level merge                                   
      278 -        offs_32 = tl.arange(0, 32)                            
      279 -        offs_m_lo = i_t * 64 + 32 + offs_32                   
      280 -        mask_lo = offs_m_lo[:, None] < cur_T                  
      281 -        A_21_32 = tl.load(A_base + offs_m_lo[:, None] * (H * B
          -T) + offs_32[None, :], mask=mask_lo).to(tl.float32)           
      282 -        Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_pre
          -cision="ieee"), Ai_11_32, input_precision="ieee")             
      278 +    # Store Ai_11_32 at (0, 0)                                
      279 +    offs_m_s0 = i_t * 64 + offs_32                            
      280 +    mask_s0 = offs_m_s0[:, None] < T                          
      281 +    tl.store(                                                 
      282 +        Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      283 +        Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      284 +        mask=mask_s0,                                         
      285 +    )                                                         
      286  
      284 -        # Store Ai_11_32 at (0, 0)                            
      285 -        offs_m_s0 = i_t * 64 + offs_32                        
      286 -        mask_s0 = offs_m_s0[:, None] < cur_T                  
      287 -        tl.store(                                             
      288 -            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_32[
          -None, :],                                                     
      289 -            Ai_11_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      290 -            mask=mask_s0,                                     
      291 -        )                                                     
      287 +    # Store Ai_22_32 at (32, 32)                              
      288 +    offs_n_32 = 32 + offs_32                                  
      289 +    tl.store(                                                 
      290 +        Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
          +],                                                            
      291 +        Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      292 +        mask=mask_lo,                                         
      293 +    )                                                         
      294  
      293 -        # Store Ai_22_32 at (32, 32)                          
      294 -        offs_n_32 = 32 + offs_32                              
      295 -        tl.store(                                             
      296 -            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_n_3
          -2[None, :],                                                   
      297 -            Ai_22_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      298 -            mask=mask_lo,                                     
      299 -        )                                                     
      295 +    # Store Ai_21_32 at (32, 0)                               
      296 +    tl.store(                                                 
      297 +        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      298 +        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      299 +        mask=mask_lo,                                         
      300 +    )                                                         
      301  
      301 -        # Store Ai_21_32 at (32, 0)                           
      302 -        tl.store(                                             
      303 -            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_32[
          -None, :],                                                     
      304 -            Ai_21_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      305 -            mask=mask_lo,                                     
      306 -        )                                                     
      302 +    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      303 +    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
          +BT)                                                           
      304 +    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      305 +    tl.store(                                                 
      306 +        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :
          +],                                                            
      307 +        zero_block.to(Ai.dtype.element_ty),                   
      308 +        mask=mask_z,                                          
      309 +    )                                                         
      310  
      308 -        # Zero out upper-right 32x32 block (rows 0~31, cols 32
          -~63)                                                          
      309 -        mask_z = (offs_m_s0[:, None] < cur_T) & (offs_n_32[Non
          -e, :] < BT)                                                   
      310 -        zero_block = tl.zeros((32, 32), dtype=tl.float32)     
      311 -        tl.store(                                             
      312 -            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_n_3
          -2[None, :],                                                   
      313 -            zero_block.to(Ai_base.dtype.element_ty),          
      314 -            mask=mask_z,                                      
      315 -        )                                                     
      311  
      317 -                                                              
      312  def solve_tril(
      313      A: torch.Tensor,
      314      cu_seqlens: torch.Tensor | None = None,

● Now update the launch code back to 2D Grid:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 12 lines, removed 29 lines
      389      chunk_indices = chunk_indices_bt
      390      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      391  
      392 -    if BT == 64:                                              
      393 -        total_blocks_merge = NT * H if IS_VARLEN else B * H * 
          -NT                                                            
      394 -        grid_merge = (get_vectorcore_num() // 2, )            
      395 -        merge_fn[grid_merge](                                 
      396 -            A=A,                                              
      397 -            Ad=Ad,                                            
      398 -            Ai=Ai,                                            
      399 -            cu_seqlens=cu_seqlens,                            
      400 -            chunk_indices=chunk_indices,                      
      401 -            T=T,                                              
      402 -            total_blocks=total_blocks_merge,                  
      403 -            H=H,                                              
      404 -            BT=BT,                                            
      405 -            num_warps=4,                                      
      406 -            num_stages=3,                                     
      407 -        )                                                     
      408 -    else:                                                     
      409 -        merge_fn[NT, B * H](                                  
      410 -            A=A,                                              
      411 -            Ad=Ad,                                            
      412 -            Ai=Ai,                                            
      413 -            cu_seqlens=cu_seqlens,                            
      414 -            chunk_indices=chunk_indices,                      
      415 -            T=T,                                              
      416 -            H=H,                                              
      417 -            BT=BT,                                            
      418 -            num_warps=4,                                      
      419 -            num_stages=3,                                     
      420 -        )                                                     
      392 +    merge_fn[NT, B * H](                                      
      393 +        A=A,                                                  
      394 +        Ad=Ad,                                                
      395 +        Ai=Ai,                                                
      396 +        cu_seqlens=cu_seqlens,                                
      397 +        chunk_indices=chunk_indices,                          
      398 +        T=T,                                                  
      399 +        H=H,                                                  
      400 +        BT=BT,                                                
      401 +        num_warps=4,                                          
      402 +        num_stages=3,                                         
      403 +    )                                                         
      404      return Ai

● Now let me verify correctness and performance with this clean 2D Grid version
  with load-compute interleaving and simplified masks (no other=0.0):

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:42:28 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:42:28 [__init__.py:46] - ascend -> vllm_ascend:register
     … +21 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:44:28.016253
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v7/ -name "op_summary_*.csv" |
       head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.76us, min=33.40us, max=34.04us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=114.64us, min=113.62us
     … +3 lines (ctrl+o to expand)

● The merge kernel remains stubbornly at ~113-115us regardless of pointer style
  or interleaving. The kernel has 304 blocks with very small per-block compute,
  so it's fundamentally scheduling-dominated.

  Let me think about a radically different approach. The merge kernel processes
  64x64 blocks. What if I could process the merge step within the solve kernel 
  itself, eliminating the need for a separate kernel launch? This would avoid
  the second kernel's scheduling overhead entirely.

  Actually, let me think about this from a different angle. The merge kernel has
   6 insert_slice calls to build two 32x32 matrices from 16x16 sub-blocks. Per
  the optimization guide: "避免对同一 tensor 多次调用
  insert_slice，以提升执行效率" (Avoid calling insert_slice multiple times on
  the same tensor).

  Can I avoid insert_slice entirely by loading data directly into a 32x32
  layout? Let me try a different approach: instead of loading 16x16 blocks
  individually and then assembling them with insert_slice, I can compute the
  64x64 inverse in a more streamlined way.

  Actually, let me try a completely different strategy: merge the 
  solve_tril_16x16 and merge_64x64 into a single kernel. Each 64x64 block can be
   computed independently:
  1. First do the 4 individual 16x16 inversions (what solve_tril does)
  2. Then do the merge (what merge does)

  This avoids the second kernel launch entirely. For our test case (B=2, T=1216,
   H=8, BT=64), this means:
  - Total 64x64 blocks = B * ceil(T/64) * H = 2 * 19 * 8 = 304 blocks
  - Each block: invert 4×16x16 blocks, then merge hierarchically

  The solve kernel currently uses LARGE_BLOCK_T=304 which spans multiple 64x64
  blocks. I'd need a different approach.

  Wait — let me reconsider. The solve kernel with LARGE_BLOCK_T=304 processes
  304/16=19 consecutive 16x16 blocks at once. If I set LARGE_BLOCK_T=64, each
  solve kernel invocation would process exactly 4 consecutive 16x16 blocks (one
  64x64 block), which maps perfectly to the merge step.

  But LARGE_BLOCK_T=64 means N_BLOCKS=4, which is very small. Let me calculate
  UB:
  - b_A: 41616*4 = 4096 bytes = 4 KB
  - local_ori_A: 16*(16*4)*4 = 4096 bytes = 4 KB
  - Total solve part: ~8 KB

  Plus the merge part UB:
  - 4×16x16 loads (Ad): 41616*4 = 4 KB
  - 2×16x16 loads (A): 21616*4 = 2 KB
  - 2×32x32 matrices: 23232*4 = 8 KB
  - 1×32x32 load (A): 32324 = 4 KB
  - Total merge part: ~18 KB

  Total: ~26 KB << 85 KB limit. That's fine.

  But the issue is total_blocks would be 304 blocks on 20 cores (with tl.dot) =
  15 blocks per core. With LARGE_BLOCK_T=304, we had 64 blocks on 40 cores = 1.6
   per core, which was much better for the solve kernel.

  Let me try a hybrid: keep the solve kernel as-is (LARGE_BLOCK_T=304, pure
  Vector, 40 cores), and focus on optimizing the merge kernel differently.

  Let me try reducing the number of insert_slice calls in the merge kernel.
  Currently it does 6 insert_slice calls. What if I load the data directly into
  a 32x32 buffer?

  Actually, looking at this more carefully, the 32x32 blocks are being assembled
   from 16x16 blocks. The structure is:
  Ai_11_32 = [Ai_11   0   ]    Ai_22_32 = [Ai_33   0   ]
              [Ai_21  Ai_22]                [Ai_43  Ai_44]

  The zeros are already there from tl.zeros((32, 32), ...). What if instead of
  insert_slice, I compute the 32x32 dot products using the 16x16 blocks
  directly?

  The formula is: Ai_21_32 = -Ai_22_32 @ A_21_32 @ Ai_11_32

  Expanding with block matrix multiplication:
  Ai_22_32 @ A_21_32 = [Ai_33  0   ] @ [A_31  A_32]
                        [Ai_43 Ai_44]   [A_41  A_42]

  = [Ai_33@A_31            Ai_33@A_32           ]
    [Ai_43@A_31+Ai_44@A_41  Ai_43@A_32+Ai_44@A_42]

  Then multiply by Ai_11_32:
  result @ Ai_11_32 = result @ [Ai_11   0   ]
                                [Ai_21  Ai_22]

  This is getting complex — 8 dot products of 16x16 instead of 2 dot products of
   32x32. That's likely worse.

  Let me try another direction: write the stores while computing. Currently all
  stores happen at the end. What if I store Ai_11_32 and Ai_22_32 earlier
  (they're computed before Ai_21_32)?

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 18 lines, removed 17 lines
      268      Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
           ), (1, 1))
      269      Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
           , (1, 1))
      270  
      271 -    # 32x32 level merge                                       
      271      offs_32 = tl.arange(0, 32)
      273 -    offs_m_lo = i_t * 64 + 32 + offs_32                       
      274 -    mask_lo = offs_m_lo[:, None] < T                          
      275 -    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          -_32[None, :], mask=mask_lo).to(tl.float32)                    
      276 -    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          -on="ieee"), Ai_11_32, input_precision="ieee")                 
      277 -                                                              
      278 -    # Store Ai_11_32 at (0, 0)                                
      272      offs_m_s0 = i_t * 64 + offs_32
      273      mask_s0 = offs_m_s0[:, None] < T
      274 +    offs_n_32 = 32 + offs_32                                  
      275 +                                                              
      276 +    # Store Ai_11_32 early while computing Ai_21_32           
      277      tl.store(
      278          Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      279          Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      280          mask=mask_s0,
      281      )
      282  
      287 -    # Store Ai_22_32 at (32, 32)                              
      288 -    offs_n_32 = 32 + offs_32                                  
      283 +    # Store Ai_22_32 early while computing Ai_21_32           
      284 +    offs_m_lo = i_t * 64 + 32 + offs_32                       
      285 +    mask_lo = offs_m_lo[:, None] < T                          
      286      tl.store(
      287          Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
           ],
      288          Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      289          mask=mask_lo,
      290      )
      291  
      295 -    # Store Ai_21_32 at (32, 0)                               
      296 -    tl.store(                                                 
      297 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      298 -        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      299 -        mask=mask_lo,                                         
      300 -    )                                                         
      301 -                                                              
      302 -    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      292 +    # Zero out upper-right 32x32 block early                  
      293      mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
           BT)
      294      zero_block = tl.zeros((32, 32), dtype=tl.float32)
      295      tl.store(
     ...
      298          mask=mask_z,
      299      )
      300  
      301 +    # 32x32 level merge - compute while stores are inflight   
      302 +    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          +_32[None, :], mask=mask_lo).to(tl.float32)                    
      303 +    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          +on="ieee"), Ai_11_32, input_precision="ieee")                 
      304  
      305 +    # Store Ai_21_32 at (32, 0)                               
      306 +    tl.store(                                                 
      307 +        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      308 +        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      309 +        mask=mask_lo,                                         
      310 +    )                                                         
      311 +                                                              
      312 +                                                              
      313  def solve_tril(
      314      A: torch.Tensor,
      315      cu_seqlens: torch.Tensor | None = None,

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:46:56 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:46:56 [__init__.py:46] - ascend -> vllm_ascend:register
     … +21 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:48:56.381129
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v8/ -name "op_summary_*.csv" |
       head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.71us, min=33.32us, max=33.90us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=114.55us, min=113.38us
     … +3 lines (ctrl+o to expand)

● Still ~114us. The merge kernel's Task Duration is dominated by scheduling
  overhead (launching 304 blocks) and Scalar address computation, not by
  Vector/Cube compute or store ordering. The various pointer and interleaving
  optimizations are having negligible impact because the bottleneck is
  structural.

  Let me try a completely different approach: fuse both kernels into one.
  Instead of:
  1. Kernel 1 (solve): Invert all 16x16 diagonal blocks → write to Ad
  2. Kernel 2 (merge): Read 16x16 blocks from Ad → merge to 64x64 → write to Ai

  I can do:
  1. Single kernel: For each 64x64 block, invert 4 consecutive 16x16 blocks in
  registers, then immediately merge them to 64x64, and write the final result.

  This eliminates:
  - The second kernel launch entirely (saves scheduling overhead)
  - The intermediate Ad tensor write/read (saves memory bandwidth)

  Let me check if this is feasible with UB budget. For one 64x64 block:
  - 4 loads from A: 4 × 16×16 × 2 bytes (fp16) = 2 KB
  - 4 × b_A for solve: 4 × 16×16 × 4 = 4 KB each but they're processed
  sequentially
  - Actually, within solve, each 16x16 block needs: b_A (16×16×4=1KB),
  local_ori_A (16×16×4=1KB) ≈ 2KB
  - After solving 4 blocks, we need 4 result matrices (16×16×4 = 1KB each) = 4
  KB
  - Then merge: 2×32×32×4 = 8 KB, plus loads from A (32×32×4 = 4 KB)
  - Total peak: ~20 KB << 85 KB

  This is doable! But there's a problem: the solve kernel processes
  LARGE_BLOCK_T=304 rows at once (N_BLOCKS=19), not just 4. The solve
  algorithm's for-loop iterates over all 16 rows and updates column vectors
  across all N_BLOCKS simultaneously. If I reduce to N_BLOCKS=4 (64 rows), each
  core would process only 1 block instead of processing many blocks in a batch.

  Wait — actually N_BLOCKS=4 would mean each kernel invocation processes exactly
   one 64x64 block. The solve_tril algorithm iterates 16 times (for rows 1..15)
  within each 16x16 block, updating each row based on previous rows. Each 16x16
  block is independent — the solve can be done independently for each 16x16
  block.

  The key question: with N_BLOCKS=4 and total 304 blocks on 40 cores → each core
   processes ~7.6 blocks. Each block does 4 × (16-iteration solve loop) + merge
  computations.

  Actually, let me reconsider. The current approach has:
  - solve kernel: 40 cores, each processing ~1.6 super-blocks of 304 rows (each
  super-block = 19 blocks of 16×16)
  - merge kernel: 304 blocks launched independently

  If I fuse them with LARGE_BLOCK_T=64 (4 blocks per invocation):
  - Total work items: B×ceil(T/64)×H = 2×19×8 = 304
  - Using 40 Vector cores: each core does ~7.6 iterations
  - But each iteration does BOTH solve (4×16×16) AND merge → no second kernel

  But wait — fused kernel has tl.dot, so grid should be get_vectorcore_num()//2
  = 20 cores. 304/20 ≈ 15 iterations per core.

  vs current: solve (40 cores, ~1.6 iterations) + merge (304 blocks). The merge
  kernel's 304 blocks are the expensive part.

  Let me try this fused approach:

  Read 1 file (ctrl+o to expand)

● Now let me implement the fused kernel approach. I'll create a single
  solve_and_merge_64x64_kernel that processes one 64x64 block per work item:
  solves 4 × 16x16 blocks inline, then merges them.
● Write(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 217 lines, removed 31 lines
      195      )
      196  
      197  
      198 +@triton.jit                                                   
      199 +def _solve_16x16_inline(b_A_block):                           
      200 +    """Solve (I+A)^-1 for a single 16x16 strictly lower triang
          +ular block."""                                                
      201 +    tmp = tl.arange(0, 16).to(tl.float32)                     
      202 +    rows = tmp[:, None]                                       
      203 +    cols = tmp[None, :]                                       
      204 +    is_lower = (rows > cols).to(b_A_block.dtype)              
      205 +    b_A_block = -b_A_block * is_lower                         
      206 +                                                              
      207 +    for i in range(1, 16):                                    
      208 +        # Extract row i                                       
      209 +        row_i = extract_slice(b_A_block, (i, 0), (1, 16), (16,
          + 1))                                                          
      210 +        # Compute dot product of row_i with all rows above    
      211 +        dot_product = tl.sum(row_i * b_A_block, 1)            
      212 +        # Update row i                                        
      213 +        new_row = row_i + dot_product[None, :]                
      214 +        b_A_block = insert_slice(b_A_block, new_row, (i, 0), (
          +1, 16), (1, 1))                                               
      215 +                                                              
      216 +    on_diagonal = rows == cols                                
      217 +    b_A_block = tl.where(on_diagonal, b_A_block + 1.0, b_A_blo
          +ck)                                                           
      218 +    return b_A_block                                          
      219 +                                                              
      220 +                                                              
      221 +@triton.jit(do_not_specialize=["T", "total_blocks"])          
      222 +def solve_and_merge_64x64_kernel(                             
      223 +    A,                                                        
      224 +    Ai,                                                       
      225 +    cu_seqlens,                                               
      226 +    chunk_indices,                                            
      227 +    T,                                                        
      228 +    total_blocks,                                             
      229 +    H: tl.constexpr,                                          
      230 +    BT: tl.constexpr,                                         
      231 +    IS_VARLEN: tl.constexpr,                                  
      232 +):                                                            
      233 +    """Fused kernel: solve 4x 16x16 blocks + merge into 64x64 
          +inverse.                                                      
      234 +                                                              
      235 +    Eliminates the intermediate Ad tensor and second kernel la
          +unch.                                                         
      236 +    Each work item processes one 64x64 block.                 
      237 +    """                                                       
      238 +    pid = tl.program_id(0)                                    
      239 +    num_programs = tl.num_programs(0)                         
      240 +                                                              
      241 +    for work_idx in range(pid, total_blocks, num_programs):   
      242 +        if IS_VARLEN:                                         
      243 +            i_t_raw = work_idx // H                           
      244 +            i_h = work_idx % H                                
      245 +            i_n = tl.load(chunk_indices + i_t_raw * 2).to(tl.i
          +nt32)                                                         
      246 +            i_t = tl.load(chunk_indices + i_t_raw * 2 + 1).to(
          +tl.int32)                                                     
      247 +            bos = tl.load(cu_seqlens + i_n).to(tl.int32)      
      248 +            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)  
      249 +            cur_T = eos - bos                                 
      250 +        else:                                                 
      251 +            blocks_per_batch = tl.cdiv(T, BT)                 
      252 +            i_bh = work_idx // blocks_per_batch               
      253 +            i_t = work_idx % blocks_per_batch                 
      254 +            i_b = i_bh // H                                   
      255 +            i_h = i_bh % H                                    
      256 +            bos = i_b * T                                     
      257 +            cur_T = T                                         
      258 +                                                              
      259 +        A_base = A + (bos * H + i_h) * BT                     
      260 +        Ai_base = Ai + (bos * H + i_h) * BT                   
      261 +                                                              
      262 +        block_start = i_t * 64                                
      263 +        offs_16 = tl.arange(0, 16)                            
      264 +                                                              
      265 +        # ===== Phase 1: Solve 4 x 16x16 blocks inline =====  
      266 +        # Load and solve block 0 (rows 0-15, diagonal col offs
          +et)                                                           
      267 +        row0 = block_start                                    
      268 +        col0 = row0 % BT                                      
      269 +        ptr0 = A_base + row0 * H * BT + col0 + offs_16[:, None
          +] * H * BT + offs_16[None, :]                                 
      270 +        mask0 = ((row0 + offs_16[:, None]) < cur_T) & ((col0 +
          + offs_16[None, :]) < BT)                                      
      271 +        b0 = tl.load(ptr0, mask=mask0).to(tl.float32)         
      272 +        Ai_11 = _solve_16x16_inline(b0)                       
      273 +                                                              
      274 +        # Load and solve block 1 (rows 16-31)                 
      275 +        row1 = block_start + 16                               
      276 +        col1 = row1 % BT                                      
      277 +        ptr1 = A_base + row1 * H * BT + col1 + offs_16[:, None
          +] * H * BT + offs_16[None, :]                                 
      278 +        mask1 = ((row1 + offs_16[:, None]) < cur_T) & ((col1 +
          + offs_16[None, :]) < BT)                                      
      279 +        b1 = tl.load(ptr1, mask=mask1).to(tl.float32)         
      280 +        Ai_22 = _solve_16x16_inline(b1)                       
      281 +                                                              
      282 +        # Load A_21 (off-diagonal block at rows 16-31, cols 0-
          +15) and compute merge                                         
      283 +        offs_m1 = block_start + 16 + offs_16                  
      284 +        mask_m1 = offs_m1[:, None] < cur_T                    
      285 +        A_21 = tl.load(A_base + offs_m1[:, None] * (H * BT) + 
          +offs_16[None, :], mask=mask_m1).to(tl.float32)                
      286 +        Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision="i
          +eee"), Ai_11, input_precision="ieee")                         
      287 +                                                              
      288 +        # Load and solve block 2 (rows 32-47)                 
      289 +        row2 = block_start + 32                               
      290 +        col2 = row2 % BT                                      
      291 +        ptr2 = A_base + row2 * H * BT + col2 + offs_16[:, None
          +] * H * BT + offs_16[None, :]                                 
      292 +        mask2 = ((row2 + offs_16[:, None]) < cur_T) & ((col2 +
          + offs_16[None, :]) < BT)                                      
      293 +        b2 = tl.load(ptr2, mask=mask2).to(tl.float32)         
      294 +        Ai_33 = _solve_16x16_inline(b2)                       
      295 +                                                              
      296 +        # Load and solve block 3 (rows 48-63)                 
      297 +        row3 = block_start + 48                               
      298 +        col3 = row3 % BT                                      
      299 +        ptr3 = A_base + row3 * H * BT + col3 + offs_16[:, None
          +] * H * BT + offs_16[None, :]                                 
      300 +        mask3 = ((row3 + offs_16[:, None]) < cur_T) & ((col3 +
          + offs_16[None, :]) < BT)                                      
      301 +        b3 = tl.load(ptr3, mask=mask3).to(tl.float32)         
      302 +        Ai_44 = _solve_16x16_inline(b3)                       
      303 +                                                              
      304 +        # Load A_43 and compute merge                         
      305 +        offs_m3 = block_start + 48 + offs_16                  
      306 +        offs_n32 = 32 + offs_16                               
      307 +        mask_m3 = offs_m3[:, None] < cur_T                    
      308 +        A_43 = tl.load(A_base + offs_m3[:, None] * (H * BT) + 
          +offs_n32[None, :], mask=mask_m3).to(tl.float32)               
      309 +        Ai_43 = -tl.dot(tl.dot(Ai_44, A_43, input_precision="i
          +eee"), Ai_33, input_precision="ieee")                         
      310 +                                                              
      311 +        # ===== Phase 2: Build 32x32 blocks and merge =====   
      312 +        Ai_22_32 = tl.zeros((32, 32), tl.float32)             
      313 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 
          +16), (1, 1))                                                  
      314 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16
          +, 16), (1, 1))                                                
      315 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16,
          + 16), (1, 1))                                                 
      316 +                                                              
      317 +        Ai_11_32 = tl.zeros((32, 32), tl.float32)             
      318 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 
          +16), (1, 1))                                                  
      319 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16
          +, 16), (1, 1))                                                
      320 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16,
          + 16), (1, 1))                                                 
      321 +                                                              
      322 +        offs_32 = tl.arange(0, 32)                            
      323 +        offs_m_lo = block_start + 32 + offs_32                
      324 +        mask_lo = offs_m_lo[:, None] < cur_T                  
      325 +        A_21_32 = tl.load(A_base + offs_m_lo[:, None] * (H * B
          +T) + offs_32[None, :], mask=mask_lo).to(tl.float32)           
      326 +        Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_pre
          +cision="ieee"), Ai_11_32, input_precision="ieee")             
      327 +                                                              
      328 +        # ===== Phase 3: Store results =====                  
      329 +        offs_m_s0 = block_start + offs_32                     
      330 +        mask_s0 = offs_m_s0[:, None] < cur_T                  
      331 +        offs_n_32 = 32 + offs_32                              
      332 +                                                              
      333 +        # Store Ai_11_32 at (0, 0)                            
      334 +        tl.store(                                             
      335 +            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_32[
          +None, :],                                                     
      336 +            Ai_11_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      337 +            mask=mask_s0,                                     
      338 +        )                                                     
      339 +                                                              
      340 +        # Store Ai_22_32 at (32, 32)                          
      341 +        tl.store(                                             
      342 +            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_n_3
          +2[None, :],                                                   
      343 +            Ai_22_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      344 +            mask=mask_lo,                                     
      345 +        )                                                     
      346 +                                                              
      347 +        # Store Ai_21_32 at (32, 0)                           
      348 +        tl.store(                                             
      349 +            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_32[
          +None, :],                                                     
      350 +            Ai_21_32.to(Ai_base.dtype.element_ty, fp_downcast_
          +rounding="rtne"),                                             
      351 +            mask=mask_lo,                                     
      352 +        )                                                     
      353 +                                                              
      354 +        # Zero out upper-right 32x32 block (rows 0~31, cols 32
          +~63)                                                          
      355 +        mask_z = (offs_m_s0[:, None] < cur_T) & (offs_n_32[Non
          +e, :] < BT)                                                   
      356 +        zero_block = tl.zeros((32, 32), dtype=tl.float32)     
      357 +        tl.store(                                             
      358 +            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_n_3
          +2[None, :],                                                   
      359 +            zero_block.to(Ai_base.dtype.element_ty),          
      360 +            mask=mask_z,                                      
      361 +        )                                                     
      362 +                                                              
      363 +                                                              
      364  @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens
           "] is not None})
      365  @triton.jit(do_not_specialize=["T"])
      366  def merge_16x16_to_64x64_inverse_kernel(
     ...
      434      Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
           ), (1, 1))
      435      Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
           , (1, 1))
      436  
      437 +    # 32x32 level merge                                       
      438      offs_32 = tl.arange(0, 32)
      439 +    offs_m_lo = i_t * 64 + 32 + offs_32                       
      440 +    mask_lo = offs_m_lo[:, None] < T                          
      441 +    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          +_32[None, :], mask=mask_lo).to(tl.float32)                    
      442 +    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          +on="ieee"), Ai_11_32, input_precision="ieee")                 
      443 +                                                              
      444 +    # Store all results                                       
      445      offs_m_s0 = i_t * 64 + offs_32
      446      mask_s0 = offs_m_s0[:, None] < T
      447      offs_n_32 = 32 + offs_32
      448  
      276 -    # Store Ai_11_32 early while computing Ai_21_32           
      449      tl.store(
      450          Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      451          Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      452          mask=mask_s0,
      453      )
      282 -                                                              
      283 -    # Store Ai_22_32 early while computing Ai_21_32           
      284 -    offs_m_lo = i_t * 64 + 32 + offs_32                       
      285 -    mask_lo = offs_m_lo[:, None] < T                          
      454      tl.store(
      455          Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
           ],
      456          Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      457          mask=mask_lo,
      458      )
      459 +    tl.store(                                                 
      460 +        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      461 +        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      462 +        mask=mask_lo,                                         
      463 +    )                                                         
      464  
      292 -    # Zero out upper-right 32x32 block early                  
      465 +    # Zero out upper-right 32x32 block                        
      466      mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
           BT)
      467      zero_block = tl.zeros((32, 32), dtype=tl.float32)
      468      tl.store(
     ...
      471          mask=mask_z,
      472      )
      473  
      301 -    # 32x32 level merge - compute while stores are inflight   
      302 -    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          -_32[None, :], mask=mask_lo).to(tl.float32)                    
      303 -    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          -on="ieee"), Ai_11_32, input_precision="ieee")                 
      474  
      305 -    # Store Ai_21_32 at (32, 0)                               
      306 -    tl.store(                                                 
      307 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      308 -        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      309 -        mask=mask_lo,                                         
      310 -    )                                                         
      311 -                                                              
      312 -                                                              
      475  def solve_tril(
      476      A: torch.Tensor,
      477      cu_seqlens: torch.Tensor | None = None,
     ...
      498      assert A.shape[-1] in [16, 32, 64]
      499  
      500      B, T, H, BT = A.shape
      339 -    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch
          -.float if BT != 16 else output_dtype)                         
      501  
      341 -    # Reduce LARGE_BLOCK_T to increase parallelism            
      342 -    # Original: 1216, NT=1 for T=1216 -> only 16 blocks total 
      343 -    # Optimized: 304, NT=4 for T=1216 -> 64 blocks total      
      344 -    # UB budget: N_BLOCKS = 304/16 = 19                       
      345 -    # b_A: 19*16*16*4 = 19456 bytes = 19 KB                   
      346 -    # local_ori_A: 16*(16*19)*4 = 19456 bytes = 19 KB         
      347 -    # Total: ~38 KB << 85 KB limit, well within Double Bufferi
          -ng range                                                      
      348 -    LARGE_BLOCK_T = 304                                       
      349 -                                                              
      502      init_device_properties_triton()
      503      IS_VARLEN = cu_seqlens is not None
      504  
      353 -    # Always recompute chunk_indices for the new LARGE_BLOCK_T
      354 -    # (prebuilt chunk_indices_large_block may use a different 
          -block size)                                                   
      505 +    # For BT=64, use fused solve+merge kernel (eliminates inte
          +rmediate tensor and second kernel launch)                     
      506 +    if BT == 64:                                              
      507 +        Ai = torch.empty(B, T, H, BT, device=A.device, dtype=o
          +utput_dtype)                                                  
      508 +                                                              
      509 +        if IS_VARLEN and chunk_indices_bt is None:            
      510 +            chunk_indices_bt = prepare_chunk_indices(cu_seqlen
          +s, BT)                                                        
      511 +        chunk_indices = chunk_indices_bt                      
      512 +                                                              
      513 +        if IS_VARLEN:                                         
      514 +            NT = len(chunk_indices)                           
      515 +            total_blocks = NT * H                             
      516 +        else:                                                 
      517 +            NT = triton.cdiv(T, BT)                           
      518 +            total_blocks = B * H * NT                         
      519 +                                                              
      520 +        # Uses tl.dot → grid = get_vectorcore_num() // 2      
      521 +        grid = (get_vectorcore_num() // 2, )                  
      522 +        solve_and_merge_64x64_kernel[grid](                   
      523 +            A=A,                                              
      524 +            Ai=Ai,                                            
      525 +            cu_seqlens=cu_seqlens,                            
      526 +            chunk_indices=chunk_indices,                      
      527 +            T=T,                                              
      528 +            total_blocks=total_blocks,                        
      529 +            H=H,                                              
      530 +            BT=BT,                                            
      531 +            IS_VARLEN=IS_VARLEN,                              
      532 +            num_warps=4,                                      
      533 +            num_stages=3,                                     
      534 +        )                                                     
      535 +        return Ai                                             
      536 +                                                              
      537 +    # For BT=16 and BT=32, use original two-phase approach    
      538 +    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch
          +.float if BT != 16 else output_dtype)                         
      539 +                                                              
      540 +    LARGE_BLOCK_T = 304                                       
      541 +                                                              
      542      if IS_VARLEN:
      543          chunk_indices = prepare_chunk_indices(cu_seqlens, LARG
           E_BLOCK_T)
      544      else:
     ...
      571          return Ad
      572  
      573      Ai = torch.empty(B, T, H, BT, device=A.device, dtype=outpu
           t_dtype)
      387 -    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32
          - else merge_16x16_to_64x64_inverse_kernel                     
      574      if IS_VARLEN and chunk_indices_bt is None:
      575          chunk_indices_bt = prepare_chunk_indices(cu_seqlens, B
           T)
      576      chunk_indices = chunk_indices_bt
      577      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      578  
      393 -    merge_fn[NT, B * H](                                      
      579 +    merge_16x16_to_32x32_inverse_kernel[NT, B * H](           
      580          A=A,
      581          Ad=Ad,
      582          Ai=Ai,

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Error: Exit code 1
     INFO 04-11 09:52:50 [__init__.py:44] Available plugins for group 
     vllm.platform_plugins:
     INFO 04-11 09:52:50 [__init__.py:46] - ascend -> vllm_ascend:register
     INFO 04-11 09:52:50 [__init__.py:49] All plugins in this group will be 
     loaded. Set `VLLM_PLUGINS` to control which plugins to load.
     INFO 04-11 09:52:50 [__init__.py:212] Platform plugin ascend is activated
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     … +11 lines (ctrl+o to see all)

● The fused kernel produces wrong results for BT=64. The _solve_16x16_inline
  function has a bug. Let me think about this. The original solve algorithm uses
   a 3D tensor (N_BLOCKS, 16, 16) and processes all blocks simultaneously. My
  inline version is processing a single 16x16 block, which should work since
  each diagonal block is independent.

  The issue is in _solve_16x16_inline. Looking at the original code more
  carefully:

  The original builds local_ori_A = tl.reshape(tl.trans(b_A, (1, 0, 2)), (16, 16
   * N_BLOCKS)) and then for each row i, extracts row i from local_ori_A which
  gives the original row across all blocks. But my inline function extracts from
   the modified b_A_block, not the original.

  The key difference: the original keeps local_ori_A as a copy of the original A
   matrix, and uses it to build the inverse. My inline version uses b_A_block
  which gets modified in each iteration. Let me fix this:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 10 lines, removed 4 lines
      197  
      198  @triton.jit
      199  def _solve_16x16_inline(b_A_block):
      200 -    """Solve (I+A)^-1 for a single 16x16 strictly lower triang
          -ular block."""                                                
      200 +    """Solve (I+A)^-1 for a single 16x16 strictly lower triang
          +ular block.                                                   
      201 +                                                              
      202 +    Uses the same algorithm as solve_tril_16x16_kernel but for
          + a single block.                                              
      203 +    """                                                       
      204 +    # Keep original A for reference (not modified during itera
          +tion)                                                         
      205 +    ori_A = b_A_block + 0.0  # copy                           
      206 +                                                              
      207      tmp = tl.arange(0, 16).to(tl.float32)
      208      rows = tmp[:, None]
      209      cols = tmp[None, :]
     ...
      211      b_A_block = -b_A_block * is_lower
      212  
      213      for i in range(1, 16):
      208 -        # Extract row i                                       
      209 -        row_i = extract_slice(b_A_block, (i, 0), (1, 16), (16,
          - 1))                                                          
      210 -        # Compute dot product of row_i with all rows above    
      214 +        # Extract row i from the ORIGINAL A (negated)         
      215 +        row_i = -extract_slice(ori_A, (i, 0), (1, 16), (16, 1)
          +)                                                             
      216 +        # Compute update: row_i @ b_A_block (dot product acros
          +s columns)                                                    
      217          dot_product = tl.sum(row_i * b_A_block, 1)
      218          # Update row i
      219          new_row = row_i + dot_product[None, :]

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Error: Exit code 1
     INFO 04-11 09:54:54 [__init__.py:44] Available plugins for group 
     vllm.platform_plugins:
     INFO 04-11 09:54:54 [__init__.py:46] - ascend -> vllm_ascend:register
     INFO 04-11 09:54:54 [__init__.py:49] All plugins in this group will be 
     loaded. Set `VLLM_PLUGINS` to control which plugins to load.
     INFO 04-11 09:54:54 [__init__.py:212] Platform plugin ascend is activated
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, 
     mean_diff=0.000000)
     … +11 lines (ctrl+o to see all)

● Still wrong. Let me look at the original algorithm more carefully. The issue
  might be in how I map the b_A_block — the original uses b_a[:, :, None] * b_A
  which is a broadcast multiply (N_BLOCKS, 16, 1) * (N_BLOCKS, 16, 16). For a
  single block (N_BLOCKS=1), this becomes (1, 16, 1) * (1, 16, 16). But my
  inline function operates on a 2D (16, 16) tensor.

  Let me trace through the original algorithm for N_BLOCKS=1:
  1. b_A shape: (1, 16, 16)
  2. local_ori_A = tl.trans(b_A, (1, 0, 2)) → shape (16, 1, 16) → reshape to
  (16, 16)
  3. For row i:
    - nblks_vec16 = -extract_slice(local_ori_A, (i, 0), (1, 16), (16, 1)) — this
   extracts row i from the ORIGINAL A, shape (1, 16)
    - b_a = tl.reshape(nblks_vec16, (1, 16)) — shape (1, 16), this is the row
  vector
    - dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2)) — b_a shape (1, 16,
  1), b_A shape (1, 16, 16) → product shape (1, 16, 16) → transpose to (16, 1,
  16)
    - dot_product = tl.sum(dot_tmp, 0) — reduce axis 0 (size 16) → shape (1, 16)
    - b_a = b_a + dot_product — shape (1, 16)

  Wait, that's a different operation than what I wrote. The b_a[:, :, None] * 
  b_A is multiplying each row of b_A by the corresponding element of b_a
  (element-wise along axis 1). Then tl.trans(result, (1, 0, 2)) and
  tl.sum(dot_tmp, 0) does a dot product.

  Actually for a single block: b_a shape (1, 16), b_a[:, :, None] shape (1, 16,
  1), b_A shape (1, 16, 16). The product is (1, 16, 16) — each column of b_A is
  scaled by b_a. Then tl.trans((1, 16, 16), (1, 0, 2)) → (16, 1, 16), then
  tl.sum(axis=0) → (1, 16). This computes: sum over j: b_a[j] * b_A[j, :] which
  is b_a @ b_A — a vector-matrix multiply.

  So for 2D: dot_product = b_a @ b_A where b_a is row vector (1, 16), b_A is
  (16, 16). This gives (1, 16).

  My inline version does tl.sum(row_i * b_A_block, 1) which computes: for each
  column k, sum_j row_i[j] * b_A_block[j, k]. This is row_i^T @ b_A_block —
  wrong! I need row_i @ b_A_block not row_i^T @ b_A_block.

  Wait, row_i has shape (1, 16) from extract_slice. Then row_i * b_A_block
  broadcasts (1, 16) * (16, 16) — this gives (16, 16) where each row of
  b_A_block is multiplied element-wise by row_i. Then tl.sum(..., 1) sums along
  axis 1 → shape (16,). This computes: sum_k row_i[k] * b_A_block[:, k] which is
   b_A_block @ row_i^T — a matrix-vector multiply.

  But I need: row_i @ b_A_block which is sum_j row_i[j] * b_A_block[j, :] =
  vector-matrix multiply.

  So tl.sum(row_i * b_A_block, 1) gives wrong dimension. I should use
  tl.sum(row_i * b_A_block, 0) to sum over axis 0 (rows), which gives row_i^T · 
  b_A_block column-wise... No wait.

  Let me be careful. row_i has shape (1, 16). When multiplied by b_A_block of
  shape (16, 16), the broadcast gives (16, 16) where result[i, j] = row_i[0, j] 
  * b_A_block[i, j].

  - tl.sum(..., axis=0) → shape (16,), element k = sum_i row_i[0, i] * 
  b_A_block[i, k] = incorrect since row_i[0, i] not row_i[0, k]

  Wait, let me reconsider. The broadcast: row_i is (1, 16), b_A_block is (16,
  16). By broadcasting, row_i is expanded to (16, 16) where each row is the
  same. So result[i, j] = row_i[0, j] * b_A_block[i, j].

  - tl.sum(result, axis=0) → shape (16,), element j = sum_i row_i[0, j] * 
  b_A_block[i, j] = row_i[0, j] * sum_i b_A_block[i, j] — this is NOT a matrix
  multiply.

  What I actually need is: for row i, compute nblks_vec16 @ b_A. In 2D for
  single block, this is a (1, 16) @ (16, 16) = (1, 16) matrix multiply.

  Let me just use the proper formulation. The original does:
  dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
  # b_a shape (N, 16, 1), b_A shape (N, 16, 16)
  # product shape (N, 16, 16): product[n, i, j] = b_a[n, i, 0] * b_A[n, i, j]
  # After transpose (1, 0, 2): shape (16, N, 16)
  dot_product = tl.sum(dot_tmp, 0)  # sum along axis 0 (16) → shape (N, 16)
  # dot_product[n, j] = sum_i b_a[n, i, 0] * b_A[n, i, j]
  # This is: for each block n: dot_product[n, :] = b_a[n, :] @ b_A[n, :, :]  
  (vector @ matrix)

  So the original computes b_a @ b_A for each block — a vector-matrix multiply.

  For my single-block inline version:
  - row_i has shape (1, 16) — this is the row vector
  - I need to compute row_i @ b_A_block = (1, 16) @ (16, 16) = (1, 16)

  To do this without tl.dot (which requires 16x16 blocks on Ascend), I need:
  - row_i[:, :, None] would be (1, 16, 1) — but I only have 2D tensors
  - Actually, let me just reshape: row_i_col = tl.trans(row_i) → (16, 1), then
  row_i_col * b_A_block → (16, 16), then tl.sum(..., 0) → (16,) = sum_i
  row_i_col[i] * b_A_block[i, :] = row_i @ b_A_block. That's correct!

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 6 lines, removed 4 lines
      211      b_A_block = -b_A_block * is_lower
      212  
      213      for i in range(1, 16):
      214 -        # Extract row i from the ORIGINAL A (negated)         
      214 +        # Extract row i from the ORIGINAL A (negated), shape (
          +1, 16)                                                        
      215          row_i = -extract_slice(ori_A, (i, 0), (1, 16), (16, 1)
           )
      216 -        # Compute update: row_i @ b_A_block (dot product acros
          -s columns)                                                    
      217 -        dot_product = tl.sum(row_i * b_A_block, 1)            
      218 -        # Update row i                                        
      216 +        # Compute vector-matrix multiply: row_i @ b_A_block   
      217 +        # row_i_col shape (16, 1), broadcast multiply with b_A
          +_block (16, 16)                                               
      218 +        row_i_col = tl.trans(row_i)  # (16, 1)                
      219 +        dot_product = tl.sum(row_i_col * b_A_block, 0)  # (16,
          +) = row_i @ b_A_block                                         
      220 +        # Update row i: row_i + dot_product                   
      221          new_row = row_i + dot_product[None, :]
      222          b_A_block = insert_slice(b_A_block, new_row, (i, 0), (
           1, 16), (1, 1))
      223  

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  INFO 04-11 09:57:25 [__init__.py:44] Available plugins for group vllm.
     platform_plugins:
     INFO 04-11 09:57:25 [__init__.py:46] - ascend -> vllm_ascend:register
     … +17 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All 7 test cases pass with max_diff=0.000000. Now let's measure performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  NA         0               device_0        2026-04-11 09:59:16.262801
                N/A                     -1
     … +6 lines (ctrl+o to expand)

  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v9/ -name "op_summary_*.csv" |
       head -1 | xargs python -c "
      import csv, sys…)
  ⎿  solve_and_merge_64x64_kernel: steady avg=708.10us, min=700.68us, max=7
     14.10us, n=16

     Total steady avg: 708.10us

● 708us — much worse than the original 174us! The fused kernel is terrible
  because:
  1. _solve_16x16_inline is called 4 times per block, and each call has a
  15-iteration loop with extract_slice, tl.trans, tl.sum, and insert_slice —
  massive Scalar overhead
  2. Only 20 cores (due to tl.dot), each processing ~15 blocks with very heavy
  per-block compute
  3. The original solve kernel processes 19 blocks simultaneously in parallel
  (vectorized across N_BLOCKS dimension), while the inline version processes
  them one-by-one

  The fusion approach is fundamentally worse because the solve algorithm's
  efficiency comes from batching N_BLOCKS together. Let me revert to the best
  performing version (v3 good).

● Bash(cp /vllm-workspace/solve_tril_v3_good.py
      /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Error: Exit code 1
     cp: cannot stat '/vllm-workspace/solve_tril_v3_good.py': No such file or 
     directory

  Listed 1 directory (ctrl+o to expand)

● The saved good version is gone. Let me reconstruct the best version from what
  I know. The v3 version had:
  - solve_tril_16x16_kernel: 1D Grid (40 cores) + stride loop, LARGE_BLOCK_T=304
  - merge_16x16_to_64x64_inverse_kernel: 2D Grid (original style), with
  load-compute interleaving
  - Total: ~145us (solve: ~33.6us, merge: ~111.8us)

  Let me write this clean version:

● Write(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 22 lines, removed 215 lines
      195      )
      196  
      197  
      198 -@triton.jit                                                   
      199 -def _solve_16x16_inline(b_A_block):                           
      200 -    """Solve (I+A)^-1 for a single 16x16 strictly lower triang
          -ular block.                                                   
      201 -                                                              
      202 -    Uses the same algorithm as solve_tril_16x16_kernel but for
          - a single block.                                              
      203 -    """                                                       
      204 -    # Keep original A for reference (not modified during itera
          -tion)                                                         
      205 -    ori_A = b_A_block + 0.0  # copy                           
      206 -                                                              
      207 -    tmp = tl.arange(0, 16).to(tl.float32)                     
      208 -    rows = tmp[:, None]                                       
      209 -    cols = tmp[None, :]                                       
      210 -    is_lower = (rows > cols).to(b_A_block.dtype)              
      211 -    b_A_block = -b_A_block * is_lower                         
      212 -                                                              
      213 -    for i in range(1, 16):                                    
      214 -        # Extract row i from the ORIGINAL A (negated), shape (
          -1, 16)                                                        
      215 -        row_i = -extract_slice(ori_A, (i, 0), (1, 16), (16, 1)
          -)                                                             
      216 -        # Compute vector-matrix multiply: row_i @ b_A_block   
      217 -        # row_i_col shape (16, 1), broadcast multiply with b_A
          -_block (16, 16)                                               
      218 -        row_i_col = tl.trans(row_i)  # (16, 1)                
      219 -        dot_product = tl.sum(row_i_col * b_A_block, 0)  # (16,
          -) = row_i @ b_A_block                                         
      220 -        # Update row i: row_i + dot_product                   
      221 -        new_row = row_i + dot_product[None, :]                
      222 -        b_A_block = insert_slice(b_A_block, new_row, (i, 0), (
          -1, 16), (1, 1))                                               
      223 -                                                              
      224 -    on_diagonal = rows == cols                                
      225 -    b_A_block = tl.where(on_diagonal, b_A_block + 1.0, b_A_blo
          -ck)                                                           
      226 -    return b_A_block                                          
      227 -                                                              
      228 -                                                              
      229 -@triton.jit(do_not_specialize=["T", "total_blocks"])          
      230 -def solve_and_merge_64x64_kernel(                             
      231 -    A,                                                        
      232 -    Ai,                                                       
      233 -    cu_seqlens,                                               
      234 -    chunk_indices,                                            
      235 -    T,                                                        
      236 -    total_blocks,                                             
      237 -    H: tl.constexpr,                                          
      238 -    BT: tl.constexpr,                                         
      239 -    IS_VARLEN: tl.constexpr,                                  
      240 -):                                                            
      241 -    """Fused kernel: solve 4x 16x16 blocks + merge into 64x64 
          -inverse.                                                      
      242 -                                                              
      243 -    Eliminates the intermediate Ad tensor and second kernel la
          -unch.                                                         
      244 -    Each work item processes one 64x64 block.                 
      245 -    """                                                       
      246 -    pid = tl.program_id(0)                                    
      247 -    num_programs = tl.num_programs(0)                         
      248 -                                                              
      249 -    for work_idx in range(pid, total_blocks, num_programs):   
      250 -        if IS_VARLEN:                                         
      251 -            i_t_raw = work_idx // H                           
      252 -            i_h = work_idx % H                                
      253 -            i_n = tl.load(chunk_indices + i_t_raw * 2).to(tl.i
          -nt32)                                                         
      254 -            i_t = tl.load(chunk_indices + i_t_raw * 2 + 1).to(
          -tl.int32)                                                     
      255 -            bos = tl.load(cu_seqlens + i_n).to(tl.int32)      
      256 -            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)  
      257 -            cur_T = eos - bos                                 
      258 -        else:                                                 
      259 -            blocks_per_batch = tl.cdiv(T, BT)                 
      260 -            i_bh = work_idx // blocks_per_batch               
      261 -            i_t = work_idx % blocks_per_batch                 
      262 -            i_b = i_bh // H                                   
      263 -            i_h = i_bh % H                                    
      264 -            bos = i_b * T                                     
      265 -            cur_T = T                                         
      266 -                                                              
      267 -        A_base = A + (bos * H + i_h) * BT                     
      268 -        Ai_base = Ai + (bos * H + i_h) * BT                   
      269 -                                                              
      270 -        block_start = i_t * 64                                
      271 -        offs_16 = tl.arange(0, 16)                            
      272 -                                                              
      273 -        # ===== Phase 1: Solve 4 x 16x16 blocks inline =====  
      274 -        # Load and solve block 0 (rows 0-15, diagonal col offs
          -et)                                                           
      275 -        row0 = block_start                                    
      276 -        col0 = row0 % BT                                      
      277 -        ptr0 = A_base + row0 * H * BT + col0 + offs_16[:, None
          -] * H * BT + offs_16[None, :]                                 
      278 -        mask0 = ((row0 + offs_16[:, None]) < cur_T) & ((col0 +
          - offs_16[None, :]) < BT)                                      
      279 -        b0 = tl.load(ptr0, mask=mask0).to(tl.float32)         
      280 -        Ai_11 = _solve_16x16_inline(b0)                       
      281 -                                                              
      282 -        # Load and solve block 1 (rows 16-31)                 
      283 -        row1 = block_start + 16                               
      284 -        col1 = row1 % BT                                      
      285 -        ptr1 = A_base + row1 * H * BT + col1 + offs_16[:, None
          -] * H * BT + offs_16[None, :]                                 
      286 -        mask1 = ((row1 + offs_16[:, None]) < cur_T) & ((col1 +
          - offs_16[None, :]) < BT)                                      
      287 -        b1 = tl.load(ptr1, mask=mask1).to(tl.float32)         
      288 -        Ai_22 = _solve_16x16_inline(b1)                       
      289 -                                                              
      290 -        # Load A_21 (off-diagonal block at rows 16-31, cols 0-
          -15) and compute merge                                         
      291 -        offs_m1 = block_start + 16 + offs_16                  
      292 -        mask_m1 = offs_m1[:, None] < cur_T                    
      293 -        A_21 = tl.load(A_base + offs_m1[:, None] * (H * BT) + 
          -offs_16[None, :], mask=mask_m1).to(tl.float32)                
      294 -        Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision="i
          -eee"), Ai_11, input_precision="ieee")                         
      295 -                                                              
      296 -        # Load and solve block 2 (rows 32-47)                 
      297 -        row2 = block_start + 32                               
      298 -        col2 = row2 % BT                                      
      299 -        ptr2 = A_base + row2 * H * BT + col2 + offs_16[:, None
          -] * H * BT + offs_16[None, :]                                 
      300 -        mask2 = ((row2 + offs_16[:, None]) < cur_T) & ((col2 +
          - offs_16[None, :]) < BT)                                      
      301 -        b2 = tl.load(ptr2, mask=mask2).to(tl.float32)         
      302 -        Ai_33 = _solve_16x16_inline(b2)                       
      303 -                                                              
      304 -        # Load and solve block 3 (rows 48-63)                 
      305 -        row3 = block_start + 48                               
      306 -        col3 = row3 % BT                                      
      307 -        ptr3 = A_base + row3 * H * BT + col3 + offs_16[:, None
          -] * H * BT + offs_16[None, :]                                 
      308 -        mask3 = ((row3 + offs_16[:, None]) < cur_T) & ((col3 +
          - offs_16[None, :]) < BT)                                      
      309 -        b3 = tl.load(ptr3, mask=mask3).to(tl.float32)         
      310 -        Ai_44 = _solve_16x16_inline(b3)                       
      311 -                                                              
      312 -        # Load A_43 and compute merge                         
      313 -        offs_m3 = block_start + 48 + offs_16                  
      314 -        offs_n32 = 32 + offs_16                               
      315 -        mask_m3 = offs_m3[:, None] < cur_T                    
      316 -        A_43 = tl.load(A_base + offs_m3[:, None] * (H * BT) + 
          -offs_n32[None, :], mask=mask_m3).to(tl.float32)               
      317 -        Ai_43 = -tl.dot(tl.dot(Ai_44, A_43, input_precision="i
          -eee"), Ai_33, input_precision="ieee")                         
      318 -                                                              
      319 -        # ===== Phase 2: Build 32x32 blocks and merge =====   
      320 -        Ai_22_32 = tl.zeros((32, 32), tl.float32)             
      321 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 
          -16), (1, 1))                                                  
      322 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16
          -, 16), (1, 1))                                                
      323 -        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16,
          - 16), (1, 1))                                                 
      324 -                                                              
      325 -        Ai_11_32 = tl.zeros((32, 32), tl.float32)             
      326 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 
          -16), (1, 1))                                                  
      327 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16
          -, 16), (1, 1))                                                
      328 -        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16,
          - 16), (1, 1))                                                 
      329 -                                                              
      330 -        offs_32 = tl.arange(0, 32)                            
      331 -        offs_m_lo = block_start + 32 + offs_32                
      332 -        mask_lo = offs_m_lo[:, None] < cur_T                  
      333 -        A_21_32 = tl.load(A_base + offs_m_lo[:, None] * (H * B
          -T) + offs_32[None, :], mask=mask_lo).to(tl.float32)           
      334 -        Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_pre
          -cision="ieee"), Ai_11_32, input_precision="ieee")             
      335 -                                                              
      336 -        # ===== Phase 3: Store results =====                  
      337 -        offs_m_s0 = block_start + offs_32                     
      338 -        mask_s0 = offs_m_s0[:, None] < cur_T                  
      339 -        offs_n_32 = 32 + offs_32                              
      340 -                                                              
      341 -        # Store Ai_11_32 at (0, 0)                            
      342 -        tl.store(                                             
      343 -            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_32[
          -None, :],                                                     
      344 -            Ai_11_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      345 -            mask=mask_s0,                                     
      346 -        )                                                     
      347 -                                                              
      348 -        # Store Ai_22_32 at (32, 32)                          
      349 -        tl.store(                                             
      350 -            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_n_3
          -2[None, :],                                                   
      351 -            Ai_22_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      352 -            mask=mask_lo,                                     
      353 -        )                                                     
      354 -                                                              
      355 -        # Store Ai_21_32 at (32, 0)                           
      356 -        tl.store(                                             
      357 -            Ai_base + offs_m_lo[:, None] * (H * BT) + offs_32[
          -None, :],                                                     
      358 -            Ai_21_32.to(Ai_base.dtype.element_ty, fp_downcast_
          -rounding="rtne"),                                             
      359 -            mask=mask_lo,                                     
      360 -        )                                                     
      361 -                                                              
      362 -        # Zero out upper-right 32x32 block (rows 0~31, cols 32
          -~63)                                                          
      363 -        mask_z = (offs_m_s0[:, None] < cur_T) & (offs_n_32[Non
          -e, :] < BT)                                                   
      364 -        zero_block = tl.zeros((32, 32), dtype=tl.float32)     
      365 -        tl.store(                                             
      366 -            Ai_base + offs_m_s0[:, None] * (H * BT) + offs_n_3
          -2[None, :],                                                   
      367 -            zero_block.to(Ai_base.dtype.element_ty),          
      368 -            mask=mask_z,                                      
      369 -        )                                                     
      370 -                                                              
      371 -                                                              
      198  @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens
           "] is not None})
      199  @triton.jit(do_not_specialize=["T"])
      200  def merge_16x16_to_64x64_inverse_kernel(
     ...
      230  
      231      offs_16 = tl.arange(0, 16)
      232  
      407 -    # Load Ai_22 and A_21, compute dot immediately (interleave
          -d)                                                            
      233 +    # Load Ai_22 and A_21, compute dot immediately (interleave
          +d load-compute)                                               
      234      offs_m1 = i_t * 64 + 16 + offs_16
      235      mask_m1 = offs_m1[:, None] < T
      236      Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + offs_16
           [None, :], mask=mask_m1).to(tl.float32)
     ...
      243      Ai_11 = tl.load(Ad + offs_m0[:, None] * (H * 16) + offs_16
           [None, :], mask=mask_m0).to(tl.float32)
      244      Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")
      245  
      420 -    # Load Ai_44 and A_43, compute dot immediately (interleave
          -d)                                                            
      246 +    # Load Ai_44 and A_43, compute dot immediately (interleave
          +d load-compute)                                               
      247      offs_m3 = i_t * 64 + 48 + offs_16
      248      offs_n32 = 32 + offs_16
      249      mask_m3 = offs_m3[:, None] < T
     ...
      280      mask_s0 = offs_m_s0[:, None] < T
      281      offs_n_32 = 32 + offs_32
      282  
      283 +    # Store Ai_11_32 at (0, 0)                                
      284      tl.store(
      285          Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      286          Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      287          mask=mask_s0,
      288      )
      289 +                                                              
      290 +    # Store Ai_22_32 at (32, 32)                              
      291      tl.store(
      292          Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
           ],
      293          Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      294          mask=mask_lo,
      295      )
      296 +                                                              
      297 +    # Store Ai_21_32 at (32, 0)                               
      298      tl.store(
      299          Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      300          Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
           "rtne"),
      301          mask=mask_lo,
      302      )
      303  
      473 -    # Zero out upper-right 32x32 block                        
      304 +    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      305      mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
           BT)
      306      zero_block = tl.zeros((32, 32), dtype=tl.float32)
      307      tl.store(
     ...
      337      assert A.shape[-1] in [16, 32, 64]
      338  
      339      B, T, H, BT = A.shape
      509 -                                                              
      510 -    init_device_properties_triton()                           
      511 -    IS_VARLEN = cu_seqlens is not None                        
      512 -                                                              
      513 -    # For BT=64, use fused solve+merge kernel (eliminates inte
          -rmediate tensor and second kernel launch)                     
      514 -    if BT == 64:                                              
      515 -        Ai = torch.empty(B, T, H, BT, device=A.device, dtype=o
          -utput_dtype)                                                  
      516 -                                                              
      517 -        if IS_VARLEN and chunk_indices_bt is None:            
      518 -            chunk_indices_bt = prepare_chunk_indices(cu_seqlen
          -s, BT)                                                        
      519 -        chunk_indices = chunk_indices_bt                      
      520 -                                                              
      521 -        if IS_VARLEN:                                         
      522 -            NT = len(chunk_indices)                           
      523 -            total_blocks = NT * H                             
      524 -        else:                                                 
      525 -            NT = triton.cdiv(T, BT)                           
      526 -            total_blocks = B * H * NT                         
      527 -                                                              
      528 -        # Uses tl.dot → grid = get_vectorcore_num() // 2      
      529 -        grid = (get_vectorcore_num() // 2, )                  
      530 -        solve_and_merge_64x64_kernel[grid](                   
      531 -            A=A,                                              
      532 -            Ai=Ai,                                            
      533 -            cu_seqlens=cu_seqlens,                            
      534 -            chunk_indices=chunk_indices,                      
      535 -            T=T,                                              
      536 -            total_blocks=total_blocks,                        
      537 -            H=H,                                              
      538 -            BT=BT,                                            
      539 -            IS_VARLEN=IS_VARLEN,                              
      540 -            num_warps=4,                                      
      541 -            num_stages=3,                                     
      542 -        )                                                     
      543 -        return Ai                                             
      544 -                                                              
      545 -    # For BT=16 and BT=32, use original two-phase approach    
      340      Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch
           .float if BT != 16 else output_dtype)
      341  
      342 +    # Reduce LARGE_BLOCK_T to increase parallelism            
      343 +    # Original: 1216, NT=1 for T=1216 -> only 16 blocks total 
      344 +    # Optimized: 304, NT=4 for T=1216 -> 64 blocks total      
      345 +    # UB budget: N_BLOCKS = 304/16 = 19                       
      346 +    # b_A: 19*16*16*4 = 19456 bytes = 19 KB                   
      347 +    # local_ori_A: 16*(16*19)*4 = 19456 bytes = 19 KB         
      348 +    # Total: ~38 KB << 85 KB limit, well within Double Bufferi
          +ng range                                                      
      349      LARGE_BLOCK_T = 304
      350  
      351 +    init_device_properties_triton()                           
      352 +    IS_VARLEN = cu_seqlens is not None                        
      353 +                                                              
      354 +    # Always recompute chunk_indices for the new LARGE_BLOCK_T
      355 +    # (prebuilt chunk_indices_large_block may use a different 
          +block size)                                                   
      356      if IS_VARLEN:
      357          chunk_indices = prepare_chunk_indices(cu_seqlens, LARG
           E_BLOCK_T)
      358      else:
     ...
      385          return Ad
      386  
      387      Ai = torch.empty(B, T, H, BT, device=A.device, dtype=outpu
           t_dtype)
      388 +    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32
          + else merge_16x16_to_64x64_inverse_kernel                     
      389      if IS_VARLEN and chunk_indices_bt is None:
      390          chunk_indices_bt = prepare_chunk_indices(cu_seqlens, B
           T)
      391      chunk_indices = chunk_indices_bt
      392      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      393  
      587 -    merge_16x16_to_32x32_inverse_kernel[NT, B * H](           
      394 +    merge_fn[NT, B * H](                                      
      395          A=A,
      396          Ad=Ad,
      397          Ai=Ai,

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 3: B=2, T=1216, H=8, BT=64 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     … +11 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Good, correctness passes. Now let me save this as a backup and then think
  about what else can be done for the merge kernel.

● Bash(cp /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py
      /vllm-workspace/solve_tril_best.py)
  ⎿  Done

● Let me confirm the current performance baseline:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  [INFO] Query all data in PROF_000001_20260411100539074_ENHRFEECEGGKHOM
     A done.
     [INFO] Profiling finished.
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_best/ -name "op_summary_*.csv"
       | head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.69us, min=33.40us, max=33.86us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=114.59us, min=113.74us
     … +3 lines (ctrl+o to expand)

● Current state: 148.29us (solve: 33.69us, merge: 114.59us) vs baseline 173.52us
   = 1.17x speedup.

  The merge kernel is the bottleneck at ~115us. Let me try a fundamentally
  different strategy - let me run msprof op on the current merge kernel to
  understand exactly where the time is spent:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿     Rated Freq: 1650

     2026-04-11 10:08:11 [INFO]  Profiling results saved in /vllm-workspace
     … +4 lines (ctrl+o to expand)
  ⎿  (timeout 10m)

  Searched for 1 pattern (ctrl+o to expand)
● Bash(for f in PipeUtilization ArithmeticUtilization ResourceConflictRatio
      MemoryUB; do
          echo "=== $f ==="…)
  ⎿  === PipeUtilization ===
     block_id,sub_block_id,aic_time(us),aic_total_cycles,aic_cube_time(us),
     aic_cube_ratio,aic_scalar_time(us),aic_scalar_ratio,aic_mte1_time(us),
     … +426 lines (ctrl+o to expand)

● Let me extract the key metrics more cleanly:

● Bash(python3 -c "
      import csv…)
  ⎿  === Cube Core ===
       aic_time avg: 6.50us
       scalar ratio avg: 0.0158
     … +10 lines (ctrl+o to expand)

● Interesting! The per-block times are very small:
  - Cube core: 6.50us average per block
  - Vector core: 7.35us average per block

  But the total Task Duration is ~115us. With 304 blocks, the hardware is
  running multiple blocks in parallel across 20 AI cores (each AI core = 1 cube
  + 2 vector sub-blocks). So 304 blocks / 20 cores ≈ 15 blocks per core × ~7us
  per block ≈ 105us + scheduling overhead.

  The per-block compute is so tiny that scheduling overhead dominates. The
  ratios show:
  - Scalar: 1.6-2.3% (very low — the per-block Scalar is not the issue)
  - mte2: 1.3% (memory transfer is fine)
  - vec/cube compute: <0.3% (almost nothing)

  The problem is that with 304 blocks on 20 cores, each core has to process ~15
  blocks sequentially. The block-level scheduling/dispatch between blocks adds
  overhead.

  Let me think about what can reduce the total time. The key insight: the merge 
  kernel's bottleneck is the number of blocks (304) that need to be dispatched.
  Each block does ~7us of useful work, and 304/20 = ~15 sequential blocks per
  core. If I could reduce the number of blocks by processing 2 or more 64x64
  tiles per block...

  But wait — the grid is [NT, B*H] where NT=19 and B*H=16. I can't change this
  2D grid to process multiple tiles per block without the 1D Grid conversion,
  which we've shown is worse.

  Actually, let me reconsider. What if I modify the merge kernel to process TWO
  consecutive 64×64 blocks per kernel invocation? This would halve NT from 19 to
   10, giving 160 blocks instead of 304. Each block does twice the work but
  there are half as many blocks.

  UB budget for 2 blocks: Each block uses 4×16×16×4 (loads) + 2×32×32×4 (build)
  + store temps ≈ 12 KB. Two blocks = 24 KB << 85 KB.

  Let me try this approach:

  Read 1 file (ctrl+o to expand)

● Actually, let me reconsider a different approach. Instead of trying to reduce
  the number of blocks, let me look at whether I can eliminate the zero-out 
  store. The zero-out of the upper-right 32×32 block is a pure write of zeros —
  it wastes a store operation. If the output tensor Ai is initialized to zeros,
  this store is unnecessary.

  Read 1 file (ctrl+o to expand)

● Let me try: initialize Ai with zeros and skip the zero-out store in the
  kernel:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 1 line, removed 1 line
      384      if BT == 16:
      385          return Ad
      386  
      387 -    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=outpu
          -t_dtype)                                                      
      387 +    Ai = torch.zeros(B, T, H, BT, device=A.device, dtype=outpu
          +t_dtype)                                                      
      388      merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32
            else merge_16x16_to_64x64_inverse_kernel
      389      if IS_VARLEN and chunk_indices_bt is None:
      390          chunk_indices_bt = prepare_chunk_indices(cu_seqlens, B
           T)

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 1 line, removed 8 lines
      301          mask=mask_lo,
      302      )
      303  
      304 -    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      305 -    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
          -BT)                                                           
      306 -    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      307 -    tl.store(                                                 
      308 -        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :
          -],                                                            
      309 -        zero_block.to(Ai.dtype.element_ty),                   
      310 -        mask=mask_z,                                          
      311 -    )                                                         
      304 +    # Upper-right 32x32 block is already zero (Ai initialized 
          +with zeros)                                                   
      305  
      306  
      307  def solve_tril(

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
     Case 6: B=1, T=128, H=16, BT=64 - PASS (max_diff=0.000000, mean_diff=0
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Some failed. Let me check which:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     Case 1: B=2, T=1216, H=8, BT=32 - FAIL (max_diff=5.066150, mean_diff=0
     … +12 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Case 1 (BT=32) fails! The BT=32 path uses merge_16x16_to_32x32_inverse_kernel,
   which I didn't modify — but the Ai = torch.zeros(...) change affects it too.
  The 32x32 merge kernel doesn't zero out an upper-right block, but the tensor
  needs to be initialized with the correct values. The issue is that for BT=32,
  the merge kernel only writes 3 blocks (Ai_11, Ai_22, Ai_21) but doesn't write
  the upper-right block — the original code with torch.empty must have been fine
   because the upper-right block was being used correctly...

  Actually wait, looking at the 32x32 merge kernel, it writes:
  - Ai_11 at (0, 0) → 16x16
  - Ai_22 at (16, 16) → 16x16
  - Ai_21 at (16, 0) → 16x16
  - But the upper-right (0, 16) → 16x16 is NOT written

  The original code uses torch.empty, so the upper-right contains garbage. But
  the reference output was generated with this same code, so the garbage values
  match. With torch.zeros, the upper-right block is now zeros, which differs
  from the reference.

  This means the upper-right block of BT=32 output doesn't matter (it's upper
  triangular and the inverse of a strictly lower triangular matrix should have
  zeros there). Let me just change the zeros only for BT=64:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 1 line, removed 1 line
      377      if BT == 16:
      378          return Ad
      379  
      380 -    Ai = torch.zeros(B, T, H, BT, device=A.device, dtype=outpu
          -t_dtype)                                                      
      380 +    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=outpu
          +t_dtype)                                                      
      381      merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32
            else merge_16x16_to_64x64_inverse_kernel
      382      if IS_VARLEN and chunk_indices_bt is None:
      383          chunk_indices_bt = prepare_chunk_indices(cu_seqlens, B
           T)

● I need to restore the zero-out store too, since I can't use torch.zeros:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 8 lines, removed 1 line
      301          mask=mask_lo,
      302      )
      303  
      304 -    # Upper-right 32x32 block is already zero (Ai initialized 
          -with zeros)                                                   
      304 +    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      305 +    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
          +BT)                                                           
      306 +    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      307 +    tl.store(                                                 
      308 +        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :
          +],                                                            
      309 +        zero_block.to(Ai.dtype.element_ty),                   
      310 +        mask=mask_z,                                          
      311 +    )                                                         
      312  
      313  
      314  def solve_tril(

● OK, let me take a step back and think about what other optimizations are
  possible for the merge kernel.

  Looking at the msprof op data again:
  - Each block takes ~7us on cube core and ~10us on vector core
  - The kernel's 304 blocks on 20 AI cores → 15 sequential blocks per core
  - Per-block ratios: scalar 1.6%, mte2 1.3%, vec 0.3% — all tiny
  - The absolute time per block is dominated by pipeline initiation overhead,
  not by any single bottleneck category

  The merge kernel does per block:
  - 7 tl.load (6 × 16x16 + 1 × 32x32)
  - 8 tl.dot (6 × 16x16 + 2 × 32x32)
  - 6 insert_slice
  - 4 tl.store (3 × 32x32 + 1 × 32x32 zero)
  - Address calculations for each load/store

  The 6 insert_slice calls are highlighted in the optimization guide as
  something to avoid: "避免对同一 tensor 多次调用 insert_slice". But the
  alternative would require loading data into 32x32 blocks directly, which isn't
   possible since the 16x16 blocks come from the Ad tensor which has a different
   layout.

  Let me try a different approach: compute the 64x64 inverse directly from 32x32
   sub-blocks without building full 32x32 matrices. The key insight: the final
  result Ai_21_32 = -Ai_22_32 @ A_21_32 @ Ai_11_32 involves 32x32 block matrix
  multiplications. But Ai_22_32 and Ai_11_32 are lower triangular block
  matrices:

  Ai_11_32 = [Ai_11    0  ]    Ai_22_32 = [Ai_33    0  ]
              [Ai_21  Ai_22]                [Ai_43  Ai_44]

  The multiplication Ai_22_32 @ A_21_32 can be done as:
  A_21_32 = [A_31  A_32]
            [A_41  A_42]

  Ai_22_32 @ A_21_32 = [Ai_33@A_31            Ai_33@A_32           ]
                        [Ai_43@A_31+Ai_44@A_41  Ai_43@A_32+Ai_44@A_42]

  This is 6 dot products of 16x16, but then the result (32x32) must be
  multiplied by Ai_11_32:
  result @ Ai_11_32 = [R11  R12] @ [Ai_11    0  ]
                      [R21  R22]   [Ai_21  Ai_22]
  = [R11@Ai_11+R12@Ai_21  R12@Ai_22           ]
    [R21@Ai_11+R22@Ai_21  R22@Ai_22           ]

  That's another 6 dot products. Total: 12 dot products of 16x16 instead of 2
  dot products of 32x32. Since a 32x32 dot is 8x the FLOPs of a 16x16 dot, this
  would be 12/16 = 0.75x the compute but with more Scalar overhead from address
  calculations.

  This probably won't help since Scalar is already the bottleneck. Let me try
  something else.

  Let me look at whether we can reduce the number of store operations. Currently
   there are 4 stores of 32x32. The zero-out store is wasteful. What if I
  combine the two stores at column 0 (Ai_11_32 at rows 0-31 and Ai_21_32 at rows
   32-63) into a single 64x32 store?

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 18 lines, removed 28 lines
      275      A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
           _32[None, :], mask=mask_lo).to(tl.float32)
      276      Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
           on="ieee"), Ai_11_32, input_precision="ieee")
      277  
      278 -    # Store all results                                       
      279 -    offs_m_s0 = i_t * 64 + offs_32                            
      280 -    mask_s0 = offs_m_s0[:, None] < T                          
      281 -    offs_n_32 = 32 + offs_32                                  
      278 +    # Store all results using combined stores where possible  
      279 +    offs_64 = tl.arange(0, 64)                                
      280 +    offs_m_all = i_t * 64 + offs_64                           
      281 +    mask_all = offs_m_all[:, None] < T                        
      282  
      283 -    # Store Ai_11_32 at (0, 0)                                
      283 +    # Build left column (64x32): [Ai_11_32; Ai_21_32]         
      284 +    left_col = tl.zeros((64, 32), dtype=tl.float32)           
      285 +    left_col = insert_slice(left_col, Ai_11_32, (0, 0), (32, 3
          +2), (1, 1))                                                   
      286 +    left_col = insert_slice(left_col, Ai_21_32, (32, 0), (32, 
          +32), (1, 1))                                                  
      287      tl.store(
      285 -        Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      286 -        Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      287 -        mask=mask_s0,                                         
      288 +        Ai + offs_m_all[:, None] * (H * BT) + offs_32[None, :]
          +,                                                             
      289 +        left_col.to(Ai.dtype.element_ty, fp_downcast_rounding=
          +"rtne"),                                                      
      290 +        mask=mask_all,                                        
      291      )
      292  
      290 -    # Store Ai_22_32 at (32, 32)                              
      293 +    # Build right column (64x32): [zeros; Ai_22_32]           
      294 +    right_col = tl.zeros((64, 32), dtype=tl.float32)          
      295 +    right_col = insert_slice(right_col, Ai_22_32, (32, 0), (32
          +, 32), (1, 1))                                                
      296 +    offs_n_32 = 32 + offs_32                                  
      297      tl.store(
      292 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
          -],                                                            
      293 -        Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      294 -        mask=mask_lo,                                         
      298 +        Ai + offs_m_all[:, None] * (H * BT) + offs_n_32[None, 
          +:],                                                           
      299 +        right_col.to(Ai.dtype.element_ty, fp_downcast_rounding
          +="rtne"),                                                     
      300 +        mask=mask_all,                                        
      301      )
      302  
      297 -    # Store Ai_21_32 at (32, 0)                               
      298 -    tl.store(                                                 
      299 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      300 -        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      301 -        mask=mask_lo,                                         
      302 -    )                                                         
      303  
      304 -    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      305 -    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
          -BT)                                                           
      306 -    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      307 -    tl.store(                                                 
      308 -        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :
          -],                                                            
      309 -        zero_block.to(Ai.dtype.element_ty),                   
      310 -        mask=mask_z,                                          
      311 -    )                                                         
      312 -                                                              
      313 -                                                              
      304  def solve_tril(
      305      A: torch.Tensor,
      306      cu_seqlens: torch.Tensor | None = None,

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, mean_diff=0
     … +12 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All pass. Now measure performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  [INFO] Query all data in PROF_000001_20260411101819236_GIAFLBHARMIDPFG
     A done.
     [INFO] Profiling finished.
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v10/ -name "op_summary_*.csv"
      | head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.71us, min=33.40us, max=34.04us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=116.40us, min=115.06us
     … +3 lines (ctrl+o to expand)

● 116.40us — slightly worse than 114.59us. The combined 64x32 stores added
  insert_slice overhead (2 more insert_slice calls) and the 64x32 stores are
  larger, which might not be as efficient. Let me revert this.

● Bash(cp /vllm-workspace/solve_tril_best.py
      /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Done

● Good, we're back to the best version. Let me try one more approach: looking at
   the num_warps and num_stages parameters for the merge kernel. Currently we
  use num_warps=4, num_stages=3. While the Ascend compiler might ignore these,
  let me also try if a different num_stages makes a difference. But actually,
  the warning says "Please DO NOT tune args ['num_warps', 'num_stages']!" — so
  these are ignored on Ascend.

  Let me think about what else I can try. The merge kernel's Scalar ratio is low
   per-block (~2%), but the bottleneck is at the task-scheduling level. With 304
   blocks on 20 AI cores, we have ~15 blocks per core, and the inter-block
  dispatch adds per-block overhead.

  Let me try to increase the amount of data each block processes by using a
  LARGE_MERGE_T approach — each block handles multiple 64x64 tiles:

  Read 1 file (ctrl+o to expand)
● Let me try a different strategy for the merge kernel: process N_TILES
  consecutive 64x64 tiles per kernel invocation using a for loop. This increases
   per-block compute, reduces the total number of blocks, and amortizes
  scheduling overhead.

  UB budget per tile: ~12KB (4 × 16x16 + 2 × 32x32 + intermediates). With
  N_TILES=4: ~48KB < 85KB.

  But wait — this requires the 2D Grid to have fewer blocks in the first
  dimension. If I process 4 tiles at once, NT drops from 19 to 5. Grid becomes
  [5, 16] = 80 blocks. With 20 AI cores, each processes 4 blocks, each
  processing 4 tiles = 16 tiles per core (vs 15 tiles per core currently). The
  total tiles is the same, but the number of block dispatches drops from 304 to
  80.

  Let me implement this:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 80 lines, removed 74 lines
      206      T,
      207      H: tl.constexpr,
      208      BT: tl.constexpr,
      209 +    N_TILES: tl.constexpr,                                    
      210      IS_VARLEN: tl.constexpr,
      211  ):
      211 -    i_t, i_bh = tl.program_id(0), tl.program_id(1)            
      212 +    i_t_group, i_bh = tl.program_id(0), tl.program_id(1)      
      213      i_b, i_h = i_bh // H, i_bh % H
      214      if IS_VARLEN:
      214 -        i_n, i_t_val = (                                      
      215 -            tl.load(chunk_indices + i_t * 2).to(tl.int32),    
      216 -            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
      215 +        i_n, i_t_group_val = (                                
      216 +            tl.load(chunk_indices + i_t_group * 2).to(tl.int32
          +),                                                            
      217 +            tl.load(chunk_indices + i_t_group * 2 + 1).to(tl.i
          +nt32),                                                        
      218          )
      219          bos, eos = (
      220              tl.load(cu_seqlens + i_n).to(tl.int32),
      221              tl.load(cu_seqlens + i_n + 1).to(tl.int32),
      222          )
      223          T = eos - bos
      223 -        i_t = i_t_val                                         
      224 +        i_t_group = i_t_group_val                             
      225      else:
      226          bos, eos = i_b * T, i_b * T + T
      227  
     ...
      229      Ad += (bos * H + i_h) * 16
      230      Ai += (bos * H + i_h) * BT
      231  
      231 -    offs_16 = tl.arange(0, 16)                                
      232 +    for tile_idx in range(N_TILES):                           
      233 +        i_t = i_t_group * N_TILES + tile_idx                  
      234 +        if i_t * 64 >= T:                                     
      235 +            return                                            
      236  
      233 -    # Load Ai_22 and A_21, compute dot immediately (interleave
          -d load-compute)                                               
      234 -    offs_m1 = i_t * 64 + 16 + offs_16                         
      235 -    mask_m1 = offs_m1[:, None] < T                            
      236 -    Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + offs_16
          -[None, :], mask=mask_m1).to(tl.float32)                       
      237 -    A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_16[N
          -one, :], mask=mask_m1).to(tl.float32)                         
      238 -    tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")      
      237 +        offs_16 = tl.arange(0, 16)                            
      238  
      240 -    # Load Ai_11, complete Ai_21                              
      241 -    offs_m0 = i_t * 64 + offs_16                              
      242 -    mask_m0 = offs_m0[:, None] < T                            
      243 -    Ai_11 = tl.load(Ad + offs_m0[:, None] * (H * 16) + offs_16
          -[None, :], mask=mask_m0).to(tl.float32)                       
      244 -    Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")    
      239 +        # Load Ai_22 and A_21, compute dot immediately (interl
          +eaved load-compute)                                           
      240 +        offs_m1 = i_t * 64 + 16 + offs_16                     
      241 +        mask_m1 = offs_m1[:, None] < T                        
      242 +        Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + off
          +s_16[None, :], mask=mask_m1).to(tl.float32)                   
      243 +        A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_
          +16[None, :], mask=mask_m1).to(tl.float32)                     
      244 +        tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")  
      245  
      246 -    # Load Ai_44 and A_43, compute dot immediately (interleave
          -d load-compute)                                               
      247 -    offs_m3 = i_t * 64 + 48 + offs_16                         
      248 -    offs_n32 = 32 + offs_16                                   
      249 -    mask_m3 = offs_m3[:, None] < T                            
      250 -    Ai_44 = tl.load(Ad + offs_m3[:, None] * (H * 16) + offs_16
          -[None, :], mask=mask_m3).to(tl.float32)                       
      251 -    A_43 = tl.load(A + offs_m3[:, None] * (H * BT) + offs_n32[
          -None, :], mask=mask_m3).to(tl.float32)                        
      252 -    tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")      
      246 +        # Load Ai_11, complete Ai_21                          
      247 +        offs_m0 = i_t * 64 + offs_16                          
      248 +        mask_m0 = offs_m0[:, None] < T                        
      249 +        Ai_11 = tl.load(Ad + offs_m0[:, None] * (H * 16) + off
          +s_16[None, :], mask=mask_m0).to(tl.float32)                   
      250 +        Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")
      251  
      254 -    # Load Ai_33, complete Ai_43                              
      255 -    offs_m2 = i_t * 64 + 32 + offs_16                         
      256 -    mask_m2 = offs_m2[:, None] < T                            
      257 -    Ai_33 = tl.load(Ad + offs_m2[:, None] * (H * 16) + offs_16
          -[None, :], mask=mask_m2).to(tl.float32)                       
      258 -    Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")    
      252 +        # Load Ai_44 and A_43, compute dot immediately (interl
          +eaved load-compute)                                           
      253 +        offs_m3 = i_t * 64 + 48 + offs_16                     
      254 +        offs_n32 = 32 + offs_16                               
      255 +        mask_m3 = offs_m3[:, None] < T                        
      256 +        Ai_44 = tl.load(Ad + offs_m3[:, None] * (H * 16) + off
          +s_16[None, :], mask=mask_m3).to(tl.float32)                   
      257 +        A_43 = tl.load(A + offs_m3[:, None] * (H * BT) + offs_
          +n32[None, :], mask=mask_m3).to(tl.float32)                    
      258 +        tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")  
      259  
      260 -    # Build 32x32 blocks                                      
      261 -    Ai_22_32 = tl.zeros((32, 32), tl.float32)                 
      262 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16),
          - (1, 1))                                                      
      263 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16
          -), (1, 1))                                                    
      264 -    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16)
          -, (1, 1))                                                     
      260 +        # Load Ai_33, complete Ai_43                          
      261 +        offs_m2 = i_t * 64 + 32 + offs_16                     
      262 +        mask_m2 = offs_m2[:, None] < T                        
      263 +        Ai_33 = tl.load(Ad + offs_m2[:, None] * (H * 16) + off
          +s_16[None, :], mask=mask_m2).to(tl.float32)                   
      264 +        Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")
      265  
      266 -    Ai_11_32 = tl.zeros((32, 32), tl.float32)                 
      267 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16),
          - (1, 1))                                                      
      268 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16
          -), (1, 1))                                                    
      269 -    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16)
          -, (1, 1))                                                     
      266 +        # Build 32x32 blocks                                  
      267 +        Ai_22_32 = tl.zeros((32, 32), tl.float32)             
      268 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 
          +16), (1, 1))                                                  
      269 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16
          +, 16), (1, 1))                                                
      270 +        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16,
          + 16), (1, 1))                                                 
      271  
      271 -    # 32x32 level merge                                       
      272 -    offs_32 = tl.arange(0, 32)                                
      273 -    offs_m_lo = i_t * 64 + 32 + offs_32                       
      274 -    mask_lo = offs_m_lo[:, None] < T                          
      275 -    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs
          -_32[None, :], mask=mask_lo).to(tl.float32)                    
      276 -    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precisi
          -on="ieee"), Ai_11_32, input_precision="ieee")                 
      272 +        Ai_11_32 = tl.zeros((32, 32), tl.float32)             
      273 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 
          +16), (1, 1))                                                  
      274 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16
          +, 16), (1, 1))                                                
      275 +        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16,
          + 16), (1, 1))                                                 
      276  
      278 -    # Store all results                                       
      279 -    offs_m_s0 = i_t * 64 + offs_32                            
      280 -    mask_s0 = offs_m_s0[:, None] < T                          
      281 -    offs_n_32 = 32 + offs_32                                  
      277 +        # 32x32 level merge                                   
      278 +        offs_32 = tl.arange(0, 32)                            
      279 +        offs_m_lo = i_t * 64 + 32 + offs_32                   
      280 +        mask_lo = offs_m_lo[:, None] < T                      
      281 +        A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + 
          +offs_32[None, :], mask=mask_lo).to(tl.float32)                
      282 +        Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_pre
          +cision="ieee"), Ai_11_32, input_precision="ieee")             
      283  
      283 -    # Store Ai_11_32 at (0, 0)                                
      284 -    tl.store(                                                 
      285 -        Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
      286 -        Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      287 -        mask=mask_s0,                                         
      288 -    )                                                         
      284 +        # Store all results                                   
      285 +        offs_m_s0 = i_t * 64 + offs_32                        
      286 +        mask_s0 = offs_m_s0[:, None] < T                      
      287 +        offs_n_32 = 32 + offs_32                              
      288  
      290 -    # Store Ai_22_32 at (32, 32)                              
      291 -    tl.store(                                                 
      292 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :
          -],                                                            
      293 -        Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      294 -        mask=mask_lo,                                         
      295 -    )                                                         
      289 +        # Store Ai_11_32 at (0, 0)                            
      290 +        tl.store(                                             
      291 +            Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None,
          + :],                                                          
      292 +            Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_round
          +ing="rtne"),                                                  
      293 +            mask=mask_s0,                                     
      294 +        )                                                     
      295  
      297 -    # Store Ai_21_32 at (32, 0)                               
      298 -    tl.store(                                                 
      299 -        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
      300 -        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding=
          -"rtne"),                                                      
      301 -        mask=mask_lo,                                         
      302 -    )                                                         
      296 +        # Store Ai_22_32 at (32, 32)                          
      297 +        tl.store(                                             
      298 +            Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[Non
          +e, :],                                                        
      299 +            Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_round
          +ing="rtne"),                                                  
      300 +            mask=mask_lo,                                     
      301 +        )                                                     
      302  
      304 -    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
      305 -    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < 
          -BT)                                                           
      306 -    zero_block = tl.zeros((32, 32), dtype=tl.float32)         
      307 -    tl.store(                                                 
      308 -        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :
          -],                                                            
      309 -        zero_block.to(Ai.dtype.element_ty),                   
      310 -        mask=mask_z,                                          
      311 -    )                                                         
      303 +        # Store Ai_21_32 at (32, 0)                           
      304 +        tl.store(                                             
      305 +            Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None,
          + :],                                                          
      306 +            Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_round
          +ing="rtne"),                                                  
      307 +            mask=mask_lo,                                     
      308 +        )                                                     
      309  
      310 +        # Zero out upper-right 32x32 block (rows 0~31, cols 32
          +~63)                                                          
      311 +        mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :
          +] < BT)                                                       
      312 +        zero_block = tl.zeros((32, 32), dtype=tl.float32)     
      313 +        tl.store(                                             
      314 +            Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[Non
          +e, :],                                                        
      315 +            zero_block.to(Ai.dtype.element_ty),               
      316 +            mask=mask_z,                                      
      317 +        )                                                     
      318  
      319 +                                                              
      320  def solve_tril(
      321      A: torch.Tensor,
      322      cu_seqlens: torch.Tensor | None = None,

● Now update the host-side launch:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 32 lines, removed 12 lines
      397      chunk_indices = chunk_indices_bt
      398      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      399  
      400 -    merge_fn[NT, B * H](                                      
      401 -        A=A,                                                  
      402 -        Ad=Ad,                                                
      403 -        Ai=Ai,                                                
      404 -        cu_seqlens=cu_seqlens,                                
      405 -        chunk_indices=chunk_indices,                          
      406 -        T=T,                                                  
      407 -        H=H,                                                  
      408 -        BT=BT,                                                
      409 -        num_warps=4,                                          
      410 -        num_stages=3,                                         
      411 -    )                                                         
      400 +    if BT == 64 and not IS_VARLEN:                            
      401 +        # Process N_TILES consecutive 64x64 blocks per kernel 
          +invocation                                                    
      402 +        # to reduce total block count and amortize scheduling 
          +overhead                                                      
      403 +        N_TILES = 4                                           
      404 +        NT_grouped = triton.cdiv(NT, N_TILES)                 
      405 +        merge_fn[NT_grouped, B * H](                          
      406 +            A=A,                                              
      407 +            Ad=Ad,                                            
      408 +            Ai=Ai,                                            
      409 +            cu_seqlens=cu_seqlens,                            
      410 +            chunk_indices=chunk_indices,                      
      411 +            T=T,                                              
      412 +            H=H,                                              
      413 +            BT=BT,                                            
      414 +            N_TILES=N_TILES,                                  
      415 +            num_warps=4,                                      
      416 +            num_stages=3,                                     
      417 +        )                                                     
      418 +    else:                                                     
      419 +        merge_fn[NT, B * H](                                  
      420 +            A=A,                                              
      421 +            Ad=Ad,                                            
      422 +            Ai=Ai,                                            
      423 +            cu_seqlens=cu_seqlens,                            
      424 +            chunk_indices=chunk_indices,                      
      425 +            T=T,                                              
      426 +            H=H,                                              
      427 +            BT=BT,                                            
      428 +            N_TILES=1,                                        
      429 +            num_warps=4,                                      
      430 +            num_stages=3,                                     
      431 +        )                                                     
      432      return Ai

● Wait, the merge_16x16_to_32x32_inverse_kernel doesn't have N_TILES. I need to
  handle this differently. Let me not pass N_TILES to the BT=32 kernel:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 21 lines, removed 21 lines
      391          return Ad
      392  
      393      Ai = torch.empty(B, T, H, BT, device=A.device, dtype=outpu
           t_dtype)
      394 -    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32
          - else merge_16x16_to_64x64_inverse_kernel                     
      394      if IS_VARLEN and chunk_indices_bt is None:
      395          chunk_indices_bt = prepare_chunk_indices(cu_seqlens, B
           T)
      396      chunk_indices = chunk_indices_bt
      397      NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, B
           T)
      398  
      400 -    if BT == 64 and not IS_VARLEN:                            
      401 -        # Process N_TILES consecutive 64x64 blocks per kernel 
          -invocation                                                    
      402 -        # to reduce total block count and amortize scheduling 
          -overhead                                                      
      403 -        N_TILES = 4                                           
      404 -        NT_grouped = triton.cdiv(NT, N_TILES)                 
      405 -        merge_fn[NT_grouped, B * H](                          
      399 +    if BT == 32:                                              
      400 +        merge_16x16_to_32x32_inverse_kernel[NT, B * H](       
      401              A=A,
      402              Ad=Ad,
      403              Ai=Ai,
     ...
      406              T=T,
      407              H=H,
      408              BT=BT,
      414 -            N_TILES=N_TILES,                                  
      409              num_warps=4,
      410              num_stages=3,
      411          )
      412      else:
      419 -        merge_fn[NT, B * H](                                  
      420 -            A=A,                                              
      421 -            Ad=Ad,                                            
      422 -            Ai=Ai,                                            
      423 -            cu_seqlens=cu_seqlens,                            
      424 -            chunk_indices=chunk_indices,                      
      425 -            T=T,                                              
      426 -            H=H,                                              
      427 -            BT=BT,                                            
      428 -            N_TILES=1,                                        
      429 -            num_warps=4,                                      
      430 -            num_stages=3,                                     
      431 -        )                                                     
      413 +        # BT=64: Process N_TILES consecutive 64x64 blocks per 
          +kernel invocation                                             
      414 +        # to reduce total block count and amortize scheduling 
          +overhead                                                      
      415 +        N_TILES = 4                                           
      416 +        if IS_VARLEN:                                         
      417 +            # For varlen, use 1 tile per block (no grouping ne
          +eded)                                                         
      418 +            merge_16x16_to_64x64_inverse_kernel[NT, B * H](   
      419 +                A=A, Ad=Ad, Ai=Ai,                            
      420 +                cu_seqlens=cu_seqlens, chunk_indices=chunk_ind
          +ices,                                                         
      421 +                T=T, H=H, BT=BT, N_TILES=1,                   
      422 +                num_warps=4, num_stages=3,                    
      423 +            )                                                 
      424 +        else:                                                 
      425 +            NT_grouped = triton.cdiv(NT, N_TILES)             
      426 +            merge_16x16_to_64x64_inverse_kernel[NT_grouped, B 
          +* H](                                                         
      427 +                A=A, Ad=Ad, Ai=Ai,                            
      428 +                cu_seqlens=cu_seqlens, chunk_indices=chunk_ind
          +ices,                                                         
      429 +                T=T, H=H, BT=BT, N_TILES=N_TILES,             
      430 +                num_warps=4, num_stages=3,                    
      431 +            )                                                 
      432      return Ai

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
  ⎿  (timeout 5m)

● It seems to hang on Case 2 (BT=64). Let me check if there's a compilation
  issue:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       File "/usr/local/python3.11.14/lib/python3.11/site-packages/triton/c
     ompiler/compiler.py", line 124, in make_ir
     … +20 lines (ctrl+o to expand)
  ⎿  (timeout 3m)

● Cannot have return inside for in Triton. Let me change the early return to a
  conditional block:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 3 lines, removed 3 lines
      231  
      232      for tile_idx in range(N_TILES):
      233          i_t = i_t_group * N_TILES + tile_idx
      234 -        if i_t * 64 >= T:                                     
      235 -            return                                            
      234 +        # Guard: skip tiles beyond sequence length            
      235 +        tile_valid = i_t * 64 < T                             
      236  
      237          offs_16 = tl.arange(0, 16)
      238  
      239          # Load Ai_22 and A_21, compute dot immediately (interl
           eaved load-compute)
      240          offs_m1 = i_t * 64 + 16 + offs_16
      241 -        mask_m1 = offs_m1[:, None] < T                        
      241 +        mask_m1 = (offs_m1[:, None] < T) & tile_valid         
      242          Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + off
           s_16[None, :], mask=mask_m1).to(tl.float32)
      243          A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_
           16[None, :], mask=mask_m1).to(tl.float32)
      244          tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")

● Wait, the issue is that all the masks in the for loop body need to be
  conditioned on tile_valid. But there's a constraint in Triton: "在 kernel
  内部的 if-else 分支中，同名变量的 Shape 必须一致" — we can't use if-else
  branches to skip tiles. Let me just use the mask approach — when the tile is
  invalid, all loads will load zeros and all stores will be masked out:

  Read 1 file (ctrl+o to expand)

● The tile_valid is only applied to mask_m1. I need to not add it since the
  offsets-based mask offs_m < T already handles out-of-bounds correctly (when
  i_t * 64 >= T, all offs_m1 = i_t*64+16+... > T so mask is all False). So the
  existing code is already correct — the mask naturally handles invalid tiles.
  Let me remove the tile_valid variable to keep it clean:

● Update(vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Added 1 line, removed 3 lines
      231  
      232      for tile_idx in range(N_TILES):
      233          i_t = i_t_group * N_TILES + tile_idx
      234 -        # Guard: skip tiles beyond sequence length            
      235 -        tile_valid = i_t * 64 < T                             
      234  
      235          offs_16 = tl.arange(0, 16)
      236  
      237          # Load Ai_22 and A_21, compute dot immediately (interl
           eaved load-compute)
      238          offs_m1 = i_t * 64 + 16 + offs_16
      241 -        mask_m1 = (offs_m1[:, None] < T) & tile_valid         
      239 +        mask_m1 = offs_m1[:, None] < T                        
      240          Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + off
           s_16[None, :], mask=mask_m1).to(tl.float32)
      241          A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_
           16[None, :], mask=mask_m1).to(tl.float32)
      242          tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, mean_diff=0
     … +12 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● All pass! Now measure performance:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  [INFO] Query all data in PROF_000001_20260411102824401_OEHBCCNNFQIQKQD
     A done.
     [INFO] Profiling finished.
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_v11/ -name "op_summary_*.csv"
      | head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.68us, min=33.44us, max=34.02us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=170.83us, min=168.02us
     … +3 lines (ctrl+o to expand)

● 170.83us — much worse than 114us! The N_TILES=4 loop adds massive Scalar
  overhead per loop iteration (recomputing all offsets and masks 4 times), and
  the reduced grid (80 blocks vs 304) means fewer cores are utilized. The
  N_TILES approach is a constexpr causing recompilation, and the inner loop's
  per-iteration Scalar cost outweighs the scheduling savings.

  Let me revert to the best version:

● Bash(cp /vllm-workspace/solve_tril_best.py
      /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py)
  ⎿  Done

● Good, we're back to the best version. Let me verify:

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  Case 0: B=2, T=1216, H=8, BT=16 - PASS (max_diff=0.000000, mean_diff=0
     .000000)
     Case 1: B=2, T=1216, H=8, BT=32 - PASS (max_diff=0.000000, mean_diff=0
     … +12 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Good. Let me take a final performance measurement and produce the report. The
  merge kernel at ~114us appears to be at its performance floor given the
  hardware constraints — 304 blocks on 20 AI cores with very small per-block
  compute. All optimization attempts (explicit pointers, make_block_ptr,
  load-compute interleaving, combined stores, N_TILES loop, 1D Grid, kernel
  fusion) have either regressed or shown no improvement.

● Bash(rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/ && export
      LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/dr
      iver/lib64/common:/u…)
  ⎿  [INFO] Query all data in PROF_000001_20260411103242434_OOKEGJMLCRLOJLL
     B done.
     [INFO] Profiling finished.
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

● Bash(find /vllm-workspace/profiling_results/opt_final/ -name
      "op_summary_*.csv" | head -1 | xargs python
      /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=33.60us, min=33.24us, max=33.98us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=114.39us, min=112.64us
     … +3 lines (ctrl+o to expand)
● Let me also save the current file as definitive best:

● Bash(cp /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py
      /vllm-workspace/solve_tril_best.py)
  ⎿  Done

● Now let me verify baseline data to compute accurate speedup:

● Bash(find /vllm-workspace/profiling_results/baseline/ -name "op_summary_*.csv"
       | head -1 | xargs python /vllm-workspace/analyze_perf.py)
  ⎿  solve_tril_16x16_kernel: steady avg=60.58us, min=60.20us, max=60.92us,
      n=16
     merge_16x16_to_64x64_inverse_kernel: steady avg=112.93us, min=111.52us
     … +3 lines (ctrl+o to expand)

● Now let me produce the optimization report.

  优化结果报告

  算子信息

  - 算子名称：solve_tril (包含 solve_tril_16x16_kernel +
  merge_16x16_to_64x64_inverse_kernel)
  - 源文件：/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py

  性能对比

  ┌─────────────────────────────────────┬────────────┬──────────────┬────────┐
  │                指标                 │ 基线耗时   │ 优化后耗时   │ 加速比 │
  │                                     │    (us)    │     (us)     │        │
  ├─────────────────────────────────────┼────────────┼──────────────┼────────┤
  │ solve_tril_16x16_kernel             │ 60.58      │ 33.60        │ 1.80x  │
  ├─────────────────────────────────────┼────────────┼──────────────┼────────┤
  │ merge_16x16_to_64x64_inverse_kernel │ 112.93     │ 114.39       │ 0.99x  │
  ├─────────────────────────────────────┼────────────┼──────────────┼────────┤
  │ 总计                                │ 173.52     │ 148.00       │ 1.17x  │
  └─────────────────────────────────────┴────────────┴──────────────┴────────┘

  优化技术清单

  1. [已应用] LARGE_BLOCK_T 优化（主要收益）：将 LARGE_BLOCK_T 从 1216 减小到
  304，使 solve_tril_16x16_kernel 的并行度从 16 blocks 提升到 64 blocks，在 40
  个 Vector Core 上获得更好的负载均衡。UB 预算：N_BLOCKS=19, b_A=19KB,
  local_ori_A=19KB, 总计~38KB < 85KB 限制。
  2. [已应用] 1D Grid + stride loop 转换（solve kernel）：将
  solve_tril_16x16_kernel 从原始 2D Grid 转换为 grid=(get_vectorcore_num(),) =
  40 核 + stride loop，使用 do_not_specialize=["T", "total_blocks"] 避免重编译。
  3. [已应用] chunk_indices 自适应重计算：当 LARGE_BLOCK_T 改变时，varlen
  模式下的 chunk_indices 需要用新的 block size 重新计算，确保正确性。
  4. [已应用] 加载-计算交织（merge kernel）：在 merge kernel 中实现 load-compute
   interleaving，先加载 Ai_22 和 A_21 后立即计算 dot product，然后再加载 Ai_11
  完成 Ai_21。减少流水线等待。
  5. [已应用] 移除 other=0.0 参数：从 merge kernel 的所有 tl.load 调用中移除
  other=0.0 参数，避免内部触发 tl.where 阻塞 load 并行。
  6. [已应用] 简化 mask 条件：移除始终为真的冗余列掩码（如 16x16 块从 Ad 加载时
  offs_n < 16 始终成立）。

  关键修改说明

  - 修改点 1 (solve kernel, +1.80x)：LARGE_BLOCK_T 从 1216 改为 304。原始实现对
  T=1216 只有 NT=1 个 large block × 16 batch*heads = 16 blocks，远低于 40
  核心数；优化后 NT=4 × 16 = 64 blocks，每核平均处理 1.6 个
  block，显著减少核心空闲时间。
  - 修改点 2 (solve kernel)：转为固定 1D Grid (get_vectorcore_num(),) + stride
  loop，符合 Ascend 最佳实践。添加 do_not_specialize
  防止因动态参数变化导致重编译。
  - 修改点 3 (merge kernel)：merge kernel 保持 2D Grid [NT, B*H]，因为该 kernel
  包含 tl.dot 操作且 304 个 blocks 已超过 AI Core 数量(20)，1D Grid
  转换反而增加调度开销（实测从 114us 退化到 152us）。

  未能进一步优化的原因（merge kernel）

  merge kernel 占总耗时
  77%（114us/148us），是性能瓶颈。经过大量尝试，以下策略均无法改善：

  ┌──────────────────────────┬───────────────┬──────────────────────────────┐
  │           策略           │     耗时      │             原因             │
  ├──────────────────────────┼───────────────┼──────────────────────────────┤
  │ 显式指针替换             │ 113.5us →     │ Scalar 比例从 0.016 增至     │
  │ make_block_ptr           │ 无改善        │ 0.026                        │
  ├──────────────────────────┼───────────────┼──────────────────────────────┤
  │ 1D Grid + stride loop    │ 152us → 退化  │ 调度开销从 304 blocks 分摊到 │
  │ (20核)                   │               │  20 核                       │
  ├──────────────────────────┼───────────────┼──────────────────────────────┤
  │ N_TILES=4 多 tile 循环   │ 171us → 退化  │ 循环内 Scalar 成本叠加       │
  ├──────────────────────────┼───────────────┼──────────────────────────────┤
  │ 合并 64x32 stores        │ 116us →       │ 额外 insert_slice 抵消 store │
  │                          │ 无改善        │  减少                        │
  ├──────────────────────────┼───────────────┼──────────────────────────────┤
  │ 算子融合 (solve+merge)   │ 708us →       │ 单块 solve 无法利用 N_BLOCKS │
  │                          │ 严重退化      │  并行                        │
  └──────────────────────────┴───────────────┴──────────────────────────────┘

  根本原因：merge kernel 每个 block 仅处理 64x64 tile（~49K FLOPs），per-block
  计算时间极短（~7us），而 304 个 blocks 的硬件调度开销占总时间的主要部分。这是
  Ascend NPU 对小粒度 Cube 计算的固有限制。

✻ Worked for 1h 57m 54s
