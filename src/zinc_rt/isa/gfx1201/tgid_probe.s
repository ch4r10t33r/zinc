.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_tgid_probe
.type zinc_rt_tgid_probe,@function

// ABI:
//   s[0:1] = output u32 pointer (one slot per workgroup)
//   s8     = workgroup_id_x (delivered iff ENABLE_SGPR_WORKGROUP_ID_X is set
//            in compute_pgm_rsrc2 AND 8 user SGPRs precede it)
//   one workitem per workgroup (num_thread_x = 1), grid = (groups,1,1)
//
// Each workgroup stores its own id at output[workgroup_id_x]. A correct
// multi-workgroup dispatch yields output = [0,1,2,...,groups-1]; if TGID is
// not delivered to s8 every workgroup collides at output[0].
zinc_rt_tgid_probe:
    s_lshl_b32 s2, s8, 2          // byte offset = workgroup_id_x * 4
    v_mov_b32_e32 v0, s2
    v_mov_b32_e32 v1, s8
    global_store_b32 v0, v1, s[0:1]
    s_waitcnt vmcnt(0)
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
