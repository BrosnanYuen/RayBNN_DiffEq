
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


	let v0: [f32; 8] = [4.1, 1.7, -0.9, 0.3, -2.0, 5.0, -1.0, 0.0];
	let mut a0 = arrayfire::Array::new(&v0, arrayfire::Dim4::new(&[8, 1, 1, 1]));

	let b0 = a0.cast::<half::f16>();
arrayfire::print_gen("b0".to_string(), &b0, Some(6));

let gg = half::f16::from_f32(2.0);

    let z = RayBNN_DiffEq::ODE::ODE45::add_one::<half::f16>(gg ,&b0);
   	arrayfire::print_gen("z".to_string(), &z, Some(6));


}
