#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_DiffEq;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_ODE() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


	let n:u64 = 10;
	let steps:u64 = 10001;



    let A_dims = arrayfire::Dim4::new(&[10,10,1,1]);
    let mut A = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_A.csv",
    	
    );
	A = arrayfire::transpose(&A, false);

    let D_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut D = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_D.csv",
    	
    );
	D = arrayfire::transpose(&D, false);

    let tspan_dims = arrayfire::Dim4::new(&[1,10001,1,1]);
    let mut tspan = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_tspan.csv",
    	
    );

    //tspan = arrayfire::transpose(&tspan, false);


    let x0_dims = arrayfire::Dim4::new(&[1,10,1,1]);
    let mut x0 = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_x0.csv",
    	
    );
	x0 = arrayfire::transpose(&x0, false);


    let xeval_dims = arrayfire::Dim4::new(&[10001,10,1,1]);
    let mut xeval = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/ODE_xeval.csv",
    	
    );
	xeval = arrayfire::transpose(&xeval, false);



	let diffeq = |t: &arrayfire::Array<f64>, x: &arrayfire::Array<f64>|  -> arrayfire::Array<f64> {
		D.clone() + arrayfire::matmul(&A, x, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE)
	};


	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0,
		tend: 100.0,
		tstep: 1E-5,
		rtol: 1E-15,
	    atol: 1.0,
		error_select: RayBNN_DiffEq::ODE::ODE45::error_type::TOTAL_ERROR
	};



	let starttime = std::time::Instant::now();



	let mut t = arrayfire::constant::<f64>(0.0,A_dims);
	let mut f = arrayfire::constant::<f64>(0.0,A_dims);
	let mut dfdt = arrayfire::constant::<f64>(0.0,A_dims);

	RayBNN_DiffEq::ODE::ODE45::solve(
		&x0
		,diffeq
		,&options
		,&mut t
		,&mut f
		,&mut dfdt
	);



	let xpred = RayBNN_DiffEq::Interpolate::Linear::run(
		&t
		,&f
		,&dfdt
		,&tspan
	);

	let elapsedtime = starttime.elapsed();

	println!("Computed {} Steps In: {:.6?}", xpred.dims()[1], elapsedtime);


	let mut relerror = xpred - xeval.clone();
    relerror = relerror/xeval;
    relerror = arrayfire::abs(&relerror);
    let (maxerr,_) =  arrayfire::max_all(&relerror);


    assert!(maxerr  <= 2E-3);

}
