.. code-block:: console

    $ pypeit_chk_for_calibs -h
    usage: pypeit_chk_for_calibs [-h] [-s SPECTROGRAPH] [-e EXTENSION]
                                 [--save_setups]
                                 root
    
    Script to check for calibrations
    
    positional arguments:
      root                  File path+root, e.g. /data/Kast/b
    
    options:
      -h, --help            show this help message and exit
      -s, --spectrograph SPECTROGRAPH
                            A valid spectrograph identifier: aat_uhrf, bok_bc,
                            gemini_flamingos1, gemini_flamingos2,
                            gemini_gmos_north_e2v, gemini_gmos_north_ham,
                            gemini_gmos_north_ham_ns, gemini_gmos_south_ham,
                            gemini_gnirs_echelle, gemini_gnirs_ifu, gtc_maat,
                            gtc_osiris, gtc_osiris_plus, jwst_nircam, jwst_nirspec,
                            keck_deimos, keck_esi, keck_hires, keck_kcrm, keck_kcwi,
                            keck_lris_blue, keck_lris_blue_orig, keck_lris_red,
                            keck_lris_red_mark4, keck_lris_red_orig, keck_mosfire,
                            keck_nires, keck_nirspec_high, keck_nirspec_high_old,
                            keck_nirspec_low, lbt_luci1, lbt_luci2, lbt_mods1b,
                            lbt_mods1r, lbt_mods2b, lbt_mods2r, ldt_deveny,
                            magellan_fire, magellan_fire_long, magellan_mage,
                            mdm_modspec, mdm_osmos_mdm4k, mdm_osmos_r4k,
                            mmt_binospec, mmt_bluechannel, mmt_mmirs, not_alfosc,
                            not_alfosc_vert, ntt_efosc2, p200_dbsp_blue,
                            p200_dbsp_red, p200_tspec, shane_kast_blue,
                            shane_kast_red, shane_kast_red_ret, soar_goodman_blue,
                            soar_goodman_red, tng_dolores, vlt_fors2, vlt_sinfoni,
                            vlt_xshooter_nir, vlt_xshooter_uvb, vlt_xshooter_vis,
                            wht_isis_blue, wht_isis_red (default: None)
      -e, --extension EXTENSION
                            File extension to use. Must include the period (e.g.,
                            ".fits") and it must be one of the allowed extensions
                            for this spectrograph. If None, root directory will be
                            searched for all files with any of the allowed
                            extensions. (default: None)
      --save_setups         If not toggled, remove setup_files/ folder and its
                            files. (default: False)
    