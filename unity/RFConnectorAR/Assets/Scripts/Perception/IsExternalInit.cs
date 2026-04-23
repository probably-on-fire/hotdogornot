// Polyfill for C# 9 `init` accessors on runtimes that do not ship
// System.Runtime.CompilerServices.IsExternalInit (Unity's .NET Standard 2.1).
// Declaring this type in user code allows the compiler to emit `init` setters.
#if !NET5_0_OR_GREATER
namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}
#endif
