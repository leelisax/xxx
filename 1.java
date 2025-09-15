//создаем список
import java.util.ArrayList;

public class Main {

  public static void main(String[] args) {

    ArrayList<String> stack = new ArrayList<String>();

// доюавляем элементы в список
    stack.add("John");
    stack.add("Jane");
    stack.add("Doe");

//выводим список
    System.out.println(stack);
    // [John, Jane, Doe]

  }
}
