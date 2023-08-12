
use arrayfire;
use RayBNN_DiffEq;

use num_traits;
use half;




const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


fn main() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);
	arrayfire::info();
	print!("Info String:\n{}", arrayfire::info_string(true));
	println!("Arrayfire version: {:?}", arrayfire::get_version());
	let (name, platform, toolkit, compute) = arrayfire::device_info();
	print!(
		"Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n",
		name, platform, toolkit, compute
	);
	println!("Revision: {}", arrayfire::get_revision());



	let diffeq = |t: &arrayfire::Array<f64>, x: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		t.clone()
	};


	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0f64,
		tend: 100.0f64,
		tstep: 0.02f64,
		rtol: 1E-5f64,
	    atol: 1E-7f64,
		normctrl: true
	};

	println!("Running");

	let starttime = std::time::Instant::now();

	let x0_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let mut t = arrayfire::constant::<f64>(0.0,x0_dims);
	let mut f = arrayfire::constant::<f64>(0.0,x0_dims);
	let mut dfdt = arrayfire::constant::<f64>(0.0,x0_dims);

	let mut x0 = arrayfire::constant::<f64>(0.0,x0_dims);

	RayBNN_DiffEq::ODE::ODE45::linear_ode_solve(
		&x0
		,diffeq
		,&options
		,&mut t
		,&mut f
		,&mut dfdt
	);

	println!("Elapsed time: {:.6?}", starttime.elapsed());

	arrayfire::print_gen("f".to_string(), &f,Some(6));

	println!("Size {}", t.dims()[0]);


}
