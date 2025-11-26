def div_vDSE_ptwise(expr_x, expr_y, expr_z):
    global div_vDSE
    div_vDSE = (
            1/Rlat*(
                expr_x.shift(longitude=-1) - expr_x
            )/del_lambda + 

            1/Rlat*(
                expr_y.shift(latitude=-1) - expr_y
            )/del_phi + 
            1*(
                expr_z.shift(isobaricInhPa=-1) - expr_z
            )/(ds_bflx['z_diff']/g)
        )
    
    return div_vDSE

